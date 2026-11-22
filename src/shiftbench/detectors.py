"""The detectors under test: everything that turns a claim and its evidence into a score.

A detector returns one number per claim, "how supported this looks", where lower means more
likely unsupported. Nothing else about it matters here: the benchmark asks what happens to a
threshold on that number when the domain changes.

| name | what it is | cost |
| --- | --- | --- |
| word-overlap | share of claim content words present in the evidence | free |
| embedding-similarity | cosine similarity of sentence embeddings | cheap |
| nli | a natural language inference model's entailment probability | medium |
| minicheck | a fact-checking model trained for exactly this judgement | medium |
| judge | a local instruction model asked to answer supported or not | slow |

The ONNX detectors load exported models from a directory, and `scripts/export_models.py`
rebuilds them from the Hub, so a reader without the directory can still reproduce the
numbers. Each export records the sequence length it was traced and parity-checked at, and
`token_limit` treats that, and the model's own position table, as ceilings.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MODELS = Path(os.environ.get("SB_MODELS", "models"))
# Used only when an export recorded no length of its own; every export here records one.
DEFAULT_MAX_TOKENS = 512
WORD = re.compile(r"[a-z0-9']+")
STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


class Detector(Protocol):
    name: str

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        """One support score per pair; lower means more likely unsupported."""


def content_words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2}


@dataclass
class WordOverlap:
    """The cheapest baseline there is, and a useful floor: does the evidence even mention it."""

    name: str = "word-overlap"

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        out = []
        for claim, passage in zip(claims, evidence, strict=True):
            words = content_words(claim)
            if not words:
                out.append(1.0)
                continue
            out.append(len(words & content_words(passage)) / len(words))
        return np.asarray(out, dtype=np.float32)


def token_limit(directory: Path, requested: int | None = None) -> int:
    """How long a sequence this export can actually be given, not how long we would like.

    Two ceilings, and the lower one wins. ``meta.json`` records the length the export was
    traced and parity-checked at, so a longer sequence is outside what was verified. The
    model's position table is harder still: RoBERTa numbers positions from pad_token_id + 1,
    so 514 rows leave 512 reachable, and one token more fails inside onnxruntime instead of
    being truncated. DeBERTa's relative attention has no such table, so there only meta.json
    applies.

    This was a hardcoded 1024. It scored the two domains whose pairs happen to be short,
    then died on the first long one, which is the worse of the two ways to be wrong: the
    scores it did produce were taken from a context longer than the export was checked at.
    """
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    limit = int(meta.get("max_length", requested or DEFAULT_MAX_TOKENS))
    if requested is not None:
        limit = min(limit, requested)
    config = json.loads((directory / "config.json").read_text())
    rows = int(config.get("max_position_embeddings", 0) or 0)
    if rows and not config.get("relative_attention"):
        reserved = 2 if str(config.get("model_type", "")).endswith("roberta") else 0
        limit = min(limit, rows - reserved)
    return limit


class OnnxEmbedding:
    """Cosine similarity between sentence embeddings of the claim and the evidence."""

    name = "embedding-similarity"

    def __init__(self, directory: Path | None = None, max_tokens: int | None = None) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        path = (directory or MODELS / "embedding").resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.max_tokens = token_limit(path, max_tokens)

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            texts, return_tensors="np", padding=True, truncation=True, max_length=self.max_tokens
        )
        feeds = {i.name: enc[i.name].astype(np.int64) for i in self.session.get_inputs() if i.name in enc}
        hidden = self.session.run(None, feeds)[0]
        mask = enc["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-6, None)
        return np.asarray(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-6, None))

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        a, b = self._embed(claims), self._embed(evidence)
        return np.asarray((a * b).sum(axis=1), dtype=np.float32)


def supported_class(directory: Path) -> int:
    """Which output of a cross-encoder means "the evidence supports the claim".

    Read from the model's own ``config.json`` rather than assumed from the shape of the
    output. This was a fixed -1, with a comment that entailment is the last class in the
    usual MNLI ordering. It is not, for the model this benchmark uses:
    cross-encoder/nli-deberta-v3-base labels 0 contradiction, 1 entailment, 2 neutral, so
    -1 read the probability of neutral and called it support.

    That one mattered more than most: the same detector name would have meant
    entailment on two domains and neutral on the three scored here. A benchmark that exists
    to measure what happens to a detector across domains would have reported its own
    indexing error as domain shift.

    A two-output model is MiniCheck's shape, whose labels config carries as "0" and "1"
    with 1 meaning supported. Anything else is refused rather than guessed at.
    """
    config = json.loads((directory / "config.json").read_text())
    labels = {int(k): str(v).lower() for k, v in config.get("id2label", {}).items()}
    named = [i for i, name in labels.items() if name.startswith("entail")]
    if named:
        return named[0]
    if len(labels) == 2:
        return 1
    raise ValueError(f"{directory}: cannot tell which of {sorted(labels.items())} means supported")


class OnnxPairClassifier:
    """An ONNX cross-encoder: the probability of the class that means supported."""

    def __init__(
        self,
        name: str,
        directory: Path,
        supported_index: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        self.name = name
        path = directory.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(path / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.supported_index = supported_class(path) if supported_index is None else supported_index
        self.max_tokens = token_limit(path, max_tokens)

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        enc = self.tokenizer(
            evidence,
            claims,
            return_tensors="np",
            padding=True,
            truncation="only_first",
            max_length=self.max_tokens,
        )
        feeds = {i.name: enc[i.name].astype(np.int64) for i in self.session.get_inputs() if i.name in enc}
        logits = self.session.run(None, feeds)[0]
        if logits.shape[-1] == 1:
            return np.asarray(1 / (1 + np.exp(-logits[:, 0])), dtype=np.float32)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        return np.asarray(probabilities[:, self.supported_index], dtype=np.float32)


class OllamaJudge:
    """A local instruction model asked whether the evidence supports the claim.

    The score is 1.0 or 0.0 from the answer word. The docstring here used to say it read
    the probability of "yes" from the token logprobs where the server returns them, which
    the code has never done.

    That matters beyond the docstring: a score with two values cannot carry a threshold at
    a promised recall, because every quantile of it is 0 or 1. So this detector cannot take
    part in the calibration comparison as it stands, and it is not in DEFAULT_DETECTORS. It
    is kept because asking a judge is what a team reaches for first, and the reason it does
    not fit here is worth being able to point at. Reading the logprobs would fix it.

    Runs locally, so the spending cap in the plan is a time cap instead: it is the slowest
    detector here.
    """

    name = "judge"

    PROMPT = (
        "Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
        "Does the evidence support the claim? Answer with one word, yes or no."
    )

    def __init__(
        self, model: str = "qwen2.5:7b", host: str = "http://127.0.0.1:11434", timeout: float = 120.0
    ) -> None:
        self.model, self.host, self.timeout = model, host, timeout

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        import httpx

        out = []
        with httpx.Client(base_url=self.host, timeout=self.timeout) as client:
            for claim, passage in zip(claims, evidence, strict=True):
                body: dict[str, Any] = {
                    "model": self.model,
                    "prompt": self.PROMPT.format(evidence=passage[:4000], claim=claim),
                    "stream": False,
                    "options": {"temperature": 0, "num_predict": 3},
                }
                answer = client.post("/api/generate", json=body).json().get("response", "")
                out.append(1.0 if answer.strip().lower().startswith("y") else 0.0)
        return np.asarray(out, dtype=np.float32)


def build(name: str, models: Path | None = None) -> Detector:
    """Make one detector by name; the ONNX ones read from ``models``."""
    root = models or MODELS
    if name == "word-overlap":
        return WordOverlap()
    if name == "embedding-similarity":
        return OnnxEmbedding(root / "embedding")
    if name == "nli":
        # Which output means support comes from the model's config, not from its shape.
        return OnnxPairClassifier("nli", root / "nli")
    if name == "minicheck":
        return OnnxPairClassifier("minicheck", root / "minicheck")
    if name == "judge":
        return OllamaJudge()
    raise KeyError(f"unknown detector {name}")


DEFAULT_DETECTORS = ("word-overlap", "embedding-similarity", "nli", "minicheck")
