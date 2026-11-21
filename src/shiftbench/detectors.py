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

The ONNX detectors load exported models from a directory. The same exports are used by
Evidence Gate, and `scripts/export_models.py` rebuilds them from the Hub, so a reader with
neither can still reproduce the numbers.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

MODELS = Path(os.environ.get("SB_MODELS", "models"))
WORD = re.compile(r"[a-z0-9']+")
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have", "in", "is",
    "it", "its", "of", "on", "or", "that", "the", "this", "to", "was", "were", "which", "with",
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


class OnnxEmbedding:
    """Cosine similarity between sentence embeddings of the claim and the evidence."""

    name = "embedding-similarity"

    def __init__(self, directory: Path | None = None, max_tokens: int = 512) -> None:
        from transformers import AutoTokenizer

        import onnxruntime as ort

        path = (directory or MODELS / "embedding").resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        self.session = ort.InferenceSession(str(path / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.max_tokens = max_tokens

    def _embed(self, texts: list[str]) -> np.ndarray:
        enc = self.tokenizer(texts, return_tensors="np", padding=True, truncation=True,
                             max_length=self.max_tokens)
        feeds = {i.name: enc[i.name].astype(np.int64) for i in self.session.get_inputs()
                 if i.name in enc}
        hidden = self.session.run(None, feeds)[0]
        mask = enc["attention_mask"][..., None].astype(np.float32)
        pooled = (hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-6, None)
        return np.asarray(pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-6, None))

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        a, b = self._embed(claims), self._embed(evidence)
        return np.asarray((a * b).sum(axis=1), dtype=np.float32)


class OnnxPairClassifier:
    """An ONNX cross-encoder: the probability of the class that means supported."""

    def __init__(self, name: str, directory: Path, supported_index: int = 0,
                 max_tokens: int = 1024) -> None:
        from transformers import AutoTokenizer

        import onnxruntime as ort

        self.name = name
        path = directory.resolve()
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "2"))
        self.session = ort.InferenceSession(str(path / "model.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.supported_index = supported_index
        self.max_tokens = max_tokens

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        enc = self.tokenizer(evidence, claims, return_tensors="np", padding=True,
                             truncation="only_first", max_length=self.max_tokens)
        feeds = {i.name: enc[i.name].astype(np.int64) for i in self.session.get_inputs()
                 if i.name in enc}
        logits = self.session.run(None, feeds)[0]
        if logits.shape[-1] == 1:
            return np.asarray(1 / (1 + np.exp(-logits[:, 0])), dtype=np.float32)
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        return np.asarray(probabilities[:, self.supported_index], dtype=np.float32)


class OllamaJudge:
    """A local instruction model asked whether the evidence supports the claim.

    The score is the probability it puts on "yes", read from the token logprobs where the
    server returns them, and otherwise 1.0 or 0.0 from the answer itself. Runs locally, so
    the spending cap in the plan is a time cap instead: it is the slowest detector here.
    """

    name = "judge"

    PROMPT = (
        "Evidence:\n{evidence}\n\nClaim: {claim}\n\n"
        "Does the evidence support the claim? Answer with one word, yes or no."
    )

    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://127.0.0.1:11434",
                 timeout: float = 120.0) -> None:
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
        # Entailment is the last class in the usual MNLI ordering.
        return OnnxPairClassifier("nli", root / "nli", supported_index=-1)
    if name == "minicheck":
        return OnnxPairClassifier("minicheck", root / "minicheck", supported_index=-1)
    if name == "judge":
        return OllamaJudge()
    raise KeyError(f"unknown detector {name}")


DEFAULT_DETECTORS = ("word-overlap", "embedding-similarity", "nli", "minicheck")
