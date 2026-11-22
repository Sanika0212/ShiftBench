"""The loaders that turn five different datasets into one shape.

Every loader is driven from local fixtures, with ``_hub`` standing in for the download, so
these run without the network. Several of them pin the shape a field arrives in rather than
the shape it is convenient to write a fixture in: MedHallu ships its Knowledge as an array of
passages and Fin-Fact its evidence as records, and a fixture that simplifies those to strings
is what let both of them be stringified into the evidence text unnoticed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from shiftbench import fetch


def write(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


def serve(monkeypatch: pytest.MonkeyPatch, files: dict[str, Path]) -> None:
    """Hand each requested filename its fixture instead of downloading it."""

    def hub(repo: str, filename: str, root: Path = fetch.RAW, revision: str = "main") -> Path:
        assert filename in files, f"unexpected file {filename}"
        return files[filename]

    monkeypatch.setattr(fetch, "_hub", hub)


def test_scientific_keys_a_claim_by_split_claim_and_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """scifact-<split>-<claim id>-<doc id>, so a cached score names the pair it came from."""
    train, validation, corpus = fetch.DOMAINS["scientific"].files
    serve(
        monkeypatch,
        {
            corpus: write(
                tmp_path / "corpus.json",
                [
                    {"doc_id": 31715818, "title": "A title", "abstract": ["one.", "two."]},
                    {"doc_id": 99, "title": "Other", "abstract": ["three."]},
                ],
            ),
            train: write(
                tmp_path / "train.json",
                [
                    {
                        "id": 0,
                        "claim": "A claim",
                        "evidences": [{"doc_id": 31715818, "label": "SUPPORT"}],
                        "cited_doc_ids": [31715818],
                    },
                    {
                        "id": 1,
                        "claim": "Another",
                        "evidences": [{"doc_id": 99, "label": "CONTRADICT"}],
                        "cited_doc_ids": [99],
                    },
                ],
            ),
            validation: write(
                tmp_path / "validation.json",
                [{"id": 7, "claim": "Third", "evidences": None, "cited_doc_ids": [99]}],
            ),
        },
    )
    frame = fetch.scientific(tmp_path)
    by_id = dict(zip(frame["claim_id"], frame["supported"], strict=True))
    assert by_id["scifact-train-0-31715818"] is True
    assert by_id["scifact-train-1-99"] is False
    # No evidence recorded is "no info", which is not support.
    assert by_id["scifact-validation-7-99"] is False
    assert frame.loc[0, "evidence"] == "A title one. two."


def test_scientific_skips_a_cited_document_it_does_not_have(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train, validation, corpus = fetch.DOMAINS["scientific"].files
    serve(
        monkeypatch,
        {
            corpus: write(tmp_path / "c.json", [{"doc_id": 1, "title": "T", "abstract": ["a."]}]),
            train: write(
                tmp_path / "t.json",
                [{"id": 0, "claim": "c", "evidences": [], "cited_doc_ids": [1, 404]}],
            ),
            validation: write(tmp_path / "v.json", []),
        },
    )
    frame = fetch.scientific(tmp_path)
    assert list(frame["claim_id"]) == ["scifact-train-0-1"]


def test_medical_makes_two_claims_per_question(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """labeled-<i>-gt and labeled-<i>-h, one supported and one not."""
    (name,) = fetch.DOMAINS["medical"].files
    serve(
        monkeypatch,
        {
            name: write(
                tmp_path / "m.json",
                [
                    {
                        "Knowledge": "Some context.",
                        "Question": "Why?",
                        "Ground Truth": "Because.",
                        "Hallucinated Answer": "Nonsense.",
                    }
                ],
            )
        },
    )
    frame = fetch.medical(tmp_path)
    assert list(frame["claim_id"]) == ["labeled-0-gt", "labeled-0-h"]
    assert list(frame["supported"]) == [True, False]
    assert frame.loc[0, "claim"] == "Why? Because."
    assert set(frame["group_id"]) == {"question-0"}


def test_legal_keys_by_split_so_the_two_splits_cannot_collide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both ContractNLI splits live under one directory, so the directory alone collides."""
    train, test = fetch.DOMAINS["legal"].files
    rows = [
        {"label": "entailment", "hypothesis": "h1", "premise": "p1", "document_id": "d1"},
        {"label": "contradiction", "hypothesis": "h2", "premise": "p2", "document_id": "d2"},
        {"label": "notmentioned", "hypothesis": "h3", "premise": "p3", "document_id": "d3"},
    ]
    serve(
        monkeypatch,
        {
            train: write(tmp_path / "train.json", rows),
            test: write(tmp_path / "test.json", rows),
        },
    )
    frame = fetch.legal(tmp_path)
    assert list(frame["claim_id"]) == [
        "contractnli-train-0",
        "contractnli-train-1",
        "contractnli-test-0",
        "contractnli-test-1",
    ]
    # "not mentioned" is neither supported nor contradicted, so it is dropped.
    assert len(frame) == 4
    assert list(frame["supported"]) == [True, False, True, False]


def test_legal_decodes_integer_labels_through_the_class_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ContractNLI arrives as 0, 1, 2 and the names live in the parquet schema.

    Read as strings, no integer starts with "e", so every legal claim came out unsupported
    and none of the "not mentioned" rows were dropped: 300 claims, 0% supported.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    train, test = fetch.DOMAINS["legal"].files
    table = pa.Table.from_pandas(
        pd.DataFrame(
            {
                "premise": ["p0", "p1", "p2"],
                "hypothesis": ["h0", "h1", "h2"],
                "label": [0, 1, 2],
            }
        )
    )
    meta = json.dumps(
        {
            "info": {
                "features": {
                    "label": {
                        "names": ["contradiction", "entailment", "neutral"],
                        "_type": "ClassLabel",
                    }
                }
            }
        }
    )
    table = table.replace_schema_metadata({"huggingface": meta})
    path = tmp_path / "legal.parquet"
    pq.write_table(table, path)
    serve(monkeypatch, {train: path, test: path})

    assert fetch._class_names(path, "label") == ["contradiction", "entailment", "neutral"]
    frame = fetch.legal(tmp_path)
    # Two rows survive per split: neutral means "not mentioned" and is dropped.
    assert len(frame) == 4
    assert list(frame["supported"]) == [False, True, False, True]


def test_general_reads_the_column_ragtruth_actually_uses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader looked for "labels" and the column is "hallucination_labels".

    So the span list was always empty and every answer came out supported: 300 claims,
    100% supported, which cannot calibrate a threshold at all.
    """
    train, test = fetch.DOMAINS["general"].files
    rows = [
        {
            "context": "Some passages.",
            "output": "A grounded answer.",
            "task_type": "Summary",
            "hallucination_labels": "[]",
            "hallucination_labels_processed": "{'evident_conflict': 0, 'baseless_info': 0}",
        },
        {
            "context": "Some passages.",
            "output": "An invented answer.",
            "task_type": "Summary",
            "hallucination_labels": '[{"start": 0, "end": 2}]',
            "hallucination_labels_processed": "{'evident_conflict': 1, 'baseless_info': 0}",
        },
    ]
    empty = write(tmp_path / "empty.json", [])
    serve(monkeypatch, {train: write(tmp_path / "train.json", rows), test: empty})
    frame = fetch.general(tmp_path)
    assert list(frame["supported"]) == [True, False]


def test_general_falls_back_to_the_processed_counts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The two columns agree on all 17,790 rows, so either settles it."""
    train, test = fetch.DOMAINS["general"].files
    rows = [
        {
            "context": "c",
            "output": "o",
            "task_type": "QA",
            "hallucination_labels_processed": {"evident_conflict": 2, "baseless_info": 0},
        }
    ]
    serve(
        monkeypatch,
        {train: write(tmp_path / "t.json", rows), test: write(tmp_path / "e.json", [])},
    )
    assert list(fetch.general(tmp_path)["supported"]) == [False]


def test_scientific_counts_a_repeated_citation_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Claim 1245 of the validation split cites document 7662395 twice, and the loader
    refused the whole domain on a duplicate claim id because of it."""
    train, validation, corpus = fetch.DOMAINS["scientific"].files
    serve(
        monkeypatch,
        {
            corpus: write(tmp_path / "c.json", [{"doc_id": 5, "title": "T", "abstract": ["a."]}]),
            train: write(
                tmp_path / "t.json",
                [{"id": 1, "claim": "c", "evidences": [], "cited_doc_ids": [5, 5]}],
            ),
            validation: write(tmp_path / "v.json", []),
        },
    )
    frame = fetch.scientific(tmp_path)
    assert list(frame["claim_id"]) == ["scifact-train-1-5"]


def test_balanced_sample_keeps_both_classes_and_respects_the_cap() -> None:
    """A quantile fitted on twelve unsupported claims is noise, so the cap samples evenly."""
    import pandas as pd

    claims = pd.DataFrame(
        {
            "claim_id": [str(i) for i in range(100)],
            "supported": [i < 70 for i in range(100)],
        }
    )
    capped = fetch.balanced_sample(claims, limit=20, seed=0)
    assert len(capped) == 20
    assert int(capped["supported"].sum()) == 10


def test_balanced_sample_returns_everything_when_under_the_cap() -> None:
    import pandas as pd

    claims = pd.DataFrame({"claim_id": ["a", "b"], "supported": [True, False]})
    assert len(fetch.balanced_sample(claims, limit=50)) == 2
    assert len(fetch.balanced_sample(claims, limit=0)) == 2


def test_text_joins_passages_instead_of_stringifying_the_container() -> None:
    """str() on MedHallu's Knowledge is numpy's repr, and that went into the evidence."""
    import numpy as np

    out = fetch.text(np.array(["First passage.", " Second passage. "], dtype=object))
    assert out == "First passage. Second passage."
    assert "[" not in out
    assert "'" not in out
    assert fetch.text(["a", ["b", "c"]]) == "a b c"
    assert fetch.text("  plain  ") == "plain"
    assert fetch.text(["a", "b"], "\n") == "a\nb"


def test_text_refuses_a_mapping() -> None:
    """Fin-Fact's evidence records have to be narrowed to a field by the loader, not guessed."""
    with pytest.raises(TypeError, match="not a mapping"):
        fetch.text({"sentence": "s", "hrefs": ["u"]})


def test_medical_joins_the_knowledge_passages(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Three passages per row is the usual shape, and 100 percent of rows had the brackets."""
    (name,) = fetch.DOMAINS["medical"].files
    serve(
        monkeypatch,
        {
            name: write(
                tmp_path / "m.json",
                [
                    {
                        "Knowledge": ["First passage.", "Second passage.", "Third passage."],
                        "Question": "Why?",
                        "Ground Truth": "Because.",
                        "Hallucinated Answer": "Nonsense.",
                    }
                ],
            )
        },
    )
    evidence = fetch.medical(tmp_path).loc[0, "evidence"]
    assert evidence == "First passage. Second passage. Third passage."


def test_financial_takes_only_the_sentences_from_its_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each Fin-Fact evidence record is {sentence, hrefs}, and the hrefs are archive URLs.

    Joining the records with str() put dict punctuation and those URLs into the evidence,
    which is noise to a grounding checker and tokens spent out of the model's 512.
    """
    (name,) = fetch.DOMAINS["financial"].files
    serve(
        monkeypatch,
        {
            name: write(
                tmp_path / "f.json",
                [
                    {
                        "claim": "Profits doubled.",
                        "label": "true",
                        "evidence": [
                            {"sentence": "Revenue rose.", "hrefs": ["https://archive.is/aaa"]},
                            {"sentence": "Costs fell.", "hrefs": ["https://archive.is/bbb"]},
                        ],
                        "issues": ["economy"],
                    }
                ],
            )
        },
    )
    row = fetch.financial(tmp_path).loc[0]
    assert row["evidence"] == "Revenue rose. Costs fell."
    assert "archive.is" not in row["evidence"]
    assert row["group_id"] == "economy"
