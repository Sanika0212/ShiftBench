"""Download each domain from its pinned source and put it in the one claim format.

Every loader answers the same three questions for a row: what is the claim, what evidence is
a reader meant to check it against, and is the claim supported by that evidence. Everything
else about the original dataset is dropped, apart from a group id that keeps claims from one
document together so a split never puts a document on both sides.

Nothing here redistributes the sources. The files land in ``data/raw`` from the Hub, and the
unified parquet stays local for the non-commercial domains.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shiftbench.domains import DOMAINS, frame

RAW = Path("data/raw")


def _hub(repo: str, filename: str, root: Path = RAW, revision: str = "main") -> Path:
    """Download one file from a dataset repository, at the revision that has it.

    The revision matters: most of these datasets keep their rows on the Hub's automatic
    Parquet branch rather than on main, and asking main for them returns 404.
    """
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo,
            filename,
            repo_type="dataset",
            revision=revision,
            local_dir=str(root / repo.split("/")[-1]),
        )
    )


def _read(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        records = pd.read_parquet(path).to_dict("records")
        return [{str(k): v for k, v in r.items()} for r in records]
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    loaded = json.loads(text)
    return loaded if isinstance(loaded, list) else [loaded]


def _class_names(path: Path, column: str) -> list[str] | None:
    """The names behind an integer label column, from the file's own schema.

    The Hub's parquet exports carry a dataset's ClassLabel names in the Arrow schema
    metadata, so an integer label can be decoded instead of guessed at. ContractNLI arrives
    as 0, 1 and 2, and the names are contradiction, entailment and neutral: the loader read
    them as strings and tested whether they started with "e", which no integer does, so
    every legal claim came out unsupported and none of the "not mentioned" rows were
    dropped. Returns None when the file does not say, and the caller then has to.
    """
    if path.suffix != ".parquet":
        return None
    import pyarrow.parquet as pq

    raw = (pq.read_schema(path).metadata or {}).get(b"huggingface")
    if not raw:
        return None
    feature = json.loads(raw.decode()).get("info", {}).get("features", {}).get(column, {})
    names = feature.get("names") if feature.get("_type") == "ClassLabel" else None
    return [str(n) for n in names] if names else None


def _first(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    raise KeyError(f"none of {names} in {sorted(row)[:12]}")


def scientific(root: Path = RAW) -> pd.DataFrame:
    """SciFact: a claim against each abstract a reviewer cited for it.

    Unsupported covers both CONTRADICT and an abstract that says nothing about the claim,
    which is what "not supported by this evidence" means here.

    The claim id is ``scifact-<split>-<claim id>-<doc id>``, which is the key Evidence Gate
    used for the same rows. ``scripts/import_evidence_gate.py`` depends on that, and checks
    the label it imports against the label here rather than trusting the id alone.
    """
    domain = DOMAINS["scientific"]
    claims_train, claims_validation, corpus_file = domain.files
    corpus = _read(_hub(domain.repo, corpus_file, root, domain.revision))
    docs = {int(r["doc_id"]): " ".join([str(r["title"]), *[str(x) for x in r["abstract"]]]) for r in corpus}
    rows = []
    for split, name in (("train", claims_train), ("validation", claims_validation)):
        claims = _read(_hub(domain.repo, name, root, domain.revision))
        for r in claims:
            labels: dict[int, str] = {}
            for ev in r["evidences"] if r["evidences"] is not None else []:
                labels.setdefault(int(ev["doc_id"]), str(ev["label"]).upper())
            # Deduplicated: claim 1245 of the validation split cites document 7662395
            # twice, and the same claim against the same document is one pair, not two.
            for doc_id in dict.fromkeys(int(d) for d in r["cited_doc_ids"]):
                passage = docs.get(int(doc_id))
                if not passage:
                    continue
                label = labels.get(int(doc_id), "NOINFO")
                rows.append(
                    {
                        "claim_id": f"scifact-{split}-{r['id']}-{doc_id}",
                        "domain": "scientific",
                        "claim": str(r["claim"]),
                        "evidence": passage,
                        "supported": label == "SUPPORT",
                        "group_id": f"doc-{doc_id}",
                        "source": "scifact",
                    }
                )
    return frame(rows)


def medical(root: Path = RAW) -> pd.DataFrame:
    """MedHallu: one grounded answer and one hallucinated answer per PubMed question."""
    domain = DOMAINS["medical"]
    rows = []
    for i, row in enumerate(_read(_hub(domain.repo, domain.files[0], root, domain.revision))):
        knowledge = str(_first(row, "Knowledge", "knowledge", "context"))
        question = str(_first(row, "Question", "question"))
        good = str(_first(row, "Ground Truth", "ground_truth", "answer"))
        bad = str(_first(row, "Hallucinated Answer", "hallucinated_answer"))
        for kind, answer, supported in (("gt", good, True), ("h", bad, False)):
            rows.append(
                {
                    # Evidence Gate's key, so its cached scores can be reused.
                    "claim_id": f"labeled-{i}-{kind}",
                    "domain": "medical",
                    "claim": f"{question} {answer}".strip(),
                    "evidence": knowledge,
                    "supported": supported,
                    "group_id": f"question-{i}",
                    "source": "medhallu",
                }
            )
    return frame(rows)


def _hallucination_count(row: dict[str, Any]) -> int:
    """RAGTruth's per-kind counts, which agree with its spans on every row of this release."""
    processed = row.get("hallucination_labels_processed")
    if isinstance(processed, str):
        import ast

        try:
            processed = ast.literal_eval(processed)
        except (ValueError, SyntaxError):
            return 0
    if not isinstance(processed, dict):
        return 0
    return sum(int(v) for v in processed.values() if str(v).lstrip("-").isdigit())


def general(root: Path = RAW) -> pd.DataFrame:
    """RAGTruth: an answer against the passages the model was given."""
    domain = DOMAINS["general"]
    rows = []
    for name in domain.files:
        for i, row in enumerate(_read(_hub(domain.repo, name, root, domain.revision))):
            passages = _first(row, "context", "passages", "source_info", "prompt")
            if isinstance(passages, list | tuple):
                passages = "\n".join(str(p) for p in passages)
            answer = str(_first(row, "response", "answer", "output"))
            # RAGTruth names these columns hallucination_labels, a list of spans, and
            # hallucination_labels_processed, a count per kind. Neither was being read, so
            # labels was always empty and every answer came out supported. The two agree on
            # all 17,790 rows of this release, which is checked rather than assumed.
            labels = (
                row.get("hallucination_labels") or row.get("labels") or row.get("hallucination_spans") or []
            )
            if isinstance(labels, str):
                labels = json.loads(labels) if labels.strip().startswith("[") else []
            flagged = len(labels) > 0 or _hallucination_count(row) > 0
            rows.append(
                {
                    "claim_id": f"ragtruth-{name.split('/')[1]}-{i}",
                    "domain": "general",
                    "claim": answer,
                    "evidence": str(passages),
                    "supported": not flagged,
                    "group_id": str(row.get("task_type") or row.get("source_id") or "answer"),
                    "source": "ragtruth",
                }
            )
    return frame(rows)


def legal(root: Path = RAW) -> pd.DataFrame:
    """ContractNLI: a hypothesis about a contract, against the contract."""
    domain = DOMAINS["legal"]
    rows = []
    for name in domain.files:
        path = _hub(domain.repo, name, root, domain.revision)
        names = _class_names(path, "label")
        for i, row in enumerate(_read(path)):
            raw = _first(row, "label", "gold_label")
            # Integer labels are decoded through the file's own ClassLabel names. Without
            # that they stringify to "0", "1" and "2", and a test for "entailment" on those
            # is false for every row.
            if names is not None and not isinstance(raw, str):
                label = names[int(raw)].lower()
            else:
                label = str(raw).lower()
            if label.startswith(("n", "2")):
                continue  # "not mentioned" is neither supported nor contradicted
            rows.append(
                {
                    # The split, not the directory: both splits live under
                    # contractnli_a, so the directory alone collides across them.
                    "claim_id": f"contractnli-{Path(name).parent.name}-{i}",
                    "domain": "legal",
                    "claim": str(_first(row, "hypothesis", "claim")),
                    "evidence": str(_first(row, "premise", "text", "document")),
                    "supported": label.startswith("e"),
                    "group_id": str(row.get("document_id") or row.get("file_name") or f"doc-{i}"),
                    "source": "contract-nli",
                }
            )
    return frame(rows)


def financial(root: Path = RAW) -> pd.DataFrame:
    """Fin-Fact: a financial claim against the evidence a fact checker used."""
    domain = DOMAINS["financial"]
    rows = []
    for i, row in enumerate(_read(_hub(domain.repo, domain.files[0], root, domain.revision))):
        label = str(_first(row, "label", "verdict")).strip().lower()
        if label.startswith("n"):
            continue  # not enough information
        evidence = _first(row, "evidence", "justification")
        if isinstance(evidence, list | tuple):
            evidence = " ".join(str(e) for e in evidence)
        rows.append(
            {
                "claim_id": f"finfact-{i}",
                "domain": "financial",
                "claim": str(_first(row, "claim")),
                "evidence": str(evidence),
                "supported": label.startswith("t"),
                "group_id": str(row.get("issues") or row.get("sector") or "finance"),
                "source": "fin-fact",
            }
        )
    return frame(rows)


LOADERS = {
    "scientific": scientific,
    "medical": medical,
    "general": general,
    "legal": legal,
    "financial": financial,
}


def balanced_sample(claims: pd.DataFrame, limit: int, seed: int = 0) -> pd.DataFrame:
    """Keep ``limit`` claims, as close to half unsupported as the domain allows.

    The benchmark runs on a laptop, so every domain is capped. Sampling evenly keeps the
    thresholds meaningful: a quantile fitted on twelve unsupported claims is noise.
    """
    if limit <= 0 or len(claims) <= limit:
        return claims.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    out = []
    for supported, group in claims.groupby("supported"):
        take = min(len(group), limit // 2 + (limit % 2 if not supported else 0))
        idx = rng.choice(len(group), take, replace=False)
        out.append(group.iloc[sorted(idx)])
    sampled = pd.concat(out).sample(frac=1, random_state=seed).reset_index(drop=True)
    return sampled.head(limit)


def build(
    domain: str, limit: int = 0, root: Path = RAW, out: Path = Path("data"), seed: int = 0
) -> pd.DataFrame:
    """Fetch one domain, unify it, cap it and write ``data/<domain>.parquet``."""
    claims = LOADERS[domain](root)
    claims = balanced_sample(claims, limit, seed)
    out.mkdir(parents=True, exist_ok=True)
    claims.to_parquet(out / f"{domain}.parquet", index=False)
    return claims
