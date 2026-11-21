"""Rebuild the ONNX detectors from the Hub, and check each export against PyTorch.

    uv run --with "optimum[onnxruntime]>=1.23" --with "torch>=2.4" \
        --with "transformers>=4.45" --with sentencepiece --with protobuf \
        python scripts/export_models.py --out models

Writes ``<out>/<name>/`` (int8) and ``<out>/<name>-fp32/`` for each cross-encoder, each
holding ``model.onnx``, ``tokenizer.json``, ``config.json`` and ``meta.json``. ``meta.json``
records the source model, the file digest and the largest difference between the exported
model's output and PyTorch's on a fixed set of pairs, so a reader can tell whether an export
is faithful before trusting a score.

Three detectors are exported. The names match ``shiftbench.detectors.build``:

| name | source | what it produces |
| --- | --- | --- |
| embedding | sentence-transformers/all-MiniLM-L6-v2 | token embeddings, mean pooled here |
| nli | cross-encoder/nli-deberta-v3-base | entailment probability |
| minicheck | lytang/MiniCheck-RoBERTa-Large | supported probability |

The cross-encoders are quantized to int8 and the detectors load the int8 directory, because
that is what ``scripts/import_evidence_gate.py`` brings in for the scientific and medical
domains. Precision has to be uniform across domains or it confounds the thing being
measured: int8 moves MiniCheck's probabilities by up to 0.0104 and the NLI model's by up to
0.0063, against a tolerance of 0.02 on recall, so a threshold fitted on int8 scores and
applied to full precision ones would shift for a reason that has nothing to do with domain.
The full precision export is kept beside it so the quantization cost stays measurable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

# Cross-encoders take (evidence, claim); the embedding model takes single texts.
PAIRS = [
    (
        "The Eiffel Tower is 330 metres tall and was completed in 1889.",
        "The tower was completed in 1889.",
    ),
    (
        "The Eiffel Tower is 330 metres tall and was completed in 1889.",
        "The tower is 500 metres tall.",
    ),
    (
        "Metformin is first-line therapy for type 2 diabetes in most adults.",
        "Metformin is recommended first for type 2 diabetes.",
    ),
    (
        "Metformin is first-line therapy for type 2 diabetes in most adults.",
        "Insulin is always the first treatment for type 2 diabetes.",
    ),
    (
        "The contract may be terminated by either party on thirty days notice.",
        "Either party can end the agreement with a month's notice.",
    ),
]

CROSS_ENCODERS = {
    "nli": "cross-encoder/nli-deberta-v3-base",
    "minicheck": "lytang/MiniCheck-RoBERTa-Large",
}
EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 512


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return np.asarray(e / e.sum(axis=-1, keepdims=True))


def keep_only(directory: Path, names: set[str]) -> None:
    """Delete everything the runtime does not read, so the directory is the artefact."""
    for f in directory.iterdir():
        if f.name in names:
            continue
        f.unlink() if f.is_file() else shutil.rmtree(f)


def torch_pair_probs(model_id: str) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
    enc = tok(
        [evidence for evidence, _ in PAIRS],
        [claim for _, claim in PAIRS],
        truncation="only_first",
        max_length=MAX_TOKENS,
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        return softmax(model(**enc).logits.float().numpy())


def torch_embedding_scores(model_id: str) -> np.ndarray:
    """Cosine similarity per pair, from PyTorch, to compare the ONNX export against."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).eval()

    def embed(texts: list[str]) -> np.ndarray:
        enc = tok(texts, padding=True, truncation=True, max_length=MAX_TOKENS,
                  return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**enc).last_hidden_state
        mask = enc["attention_mask"][..., None].float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        pooled = pooled / pooled.norm(dim=1, keepdim=True).clamp(min=1e-6)
        return np.asarray(pooled.numpy())

    a = embed([claim for _, claim in PAIRS])
    b = embed([evidence for evidence, _ in PAIRS])
    return np.asarray((a * b).sum(axis=1))


def onnx_scores(name: str, directory: Path) -> np.ndarray:
    """Score the fixed pairs with the exported model, through the benchmark's own loaders."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from shiftbench.detectors import OnnxEmbedding, OnnxPairClassifier

    claims = [claim for _, claim in PAIRS]
    evidence = [evidence for evidence, _ in PAIRS]
    if name == "embedding":
        return OnnxEmbedding(directory).score(claims, evidence)
    return OnnxPairClassifier(name, directory, supported_index=-1).score(claims, evidence)


def export_cross_encoder(name: str, model_id: str, out: Path) -> dict[str, object]:
    """Export to full precision, then quantize to int8; the detector reads the int8 one."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    fp32, int8 = out / f"{name}-fp32", out / name
    for d in (fp32, int8):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
    started = time.time()
    ORTModelForSequenceClassification.from_pretrained(model_id, export=True).save_pretrained(fp32)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(fp32)
    if not (fp32 / "tokenizer.json").exists():
        raise SystemExit(f"{model_id}: no fast tokenizer (tokenizer.json) was written")
    keep_only(fp32, {"model.onnx", "model.onnx_data", "tokenizer.json", "config.json"})
    # Per-channel int8 on MatMul only. Per-tensor quantization, the default, breaks
    # DeBERTa-v3: its entailment probabilities moved by as much as 0.99.
    quantize_dynamic(
        fp32 / "model.onnx",
        int8 / "model.onnx",
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        per_channel=True,
        reduce_range=True,
    )
    for f in ("tokenizer.json", "config.json"):
        shutil.copy(fp32 / f, int8 / f)
    # Entailment is the last class in the usual ordering, which is what the detector reads.
    reference = torch_pair_probs(model_id)[:, -1]
    report = finish(name, model_id, int8, reference, started, precision="int8")
    report["fp32"] = finish(f"{name}-fp32", model_id, fp32, reference, started, precision="fp32")
    return report


def export_embedding(out: Path, model_id: str = EMBEDDING) -> dict[str, object]:
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    directory = out / "embedding"
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    started = time.time()
    ORTModelForFeatureExtraction.from_pretrained(model_id, export=True).save_pretrained(directory)
    AutoTokenizer.from_pretrained(model_id).save_pretrained(directory)
    keep_only(directory, {"model.onnx", "model.onnx_data", "tokenizer.json", "config.json"})
    return finish("embedding", model_id, directory, torch_embedding_scores(model_id), started)


def finish(name: str, model_id: str, directory: Path, reference: np.ndarray,
           started: float, precision: str = "fp32") -> dict[str, object]:
    """Write meta.json, then measure the export against PyTorch and write it again."""
    meta: dict[str, object] = {
        "name": name,
        "source": model_id,
        "precision": precision,
        "max_length": MAX_TOKENS,
        "sha256": sha256(directory / "model.onnx"),
        "bytes": (directory / "model.onnx").stat().st_size,
        "exported": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export_seconds": round(time.time() - started, 1),
    }
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    scored = onnx_scores(name.removesuffix("-fp32"), directory)
    meta["max_abs_diff_vs_torch"] = float(np.abs(scored - reference).max())
    (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--only", action="append", choices=["embedding", "nli", "minicheck"],
                    default=None)
    ap.add_argument("--tolerance", type=float, default=1e-4,
                    help="fail if a full precision export differs from PyTorch by more than this")
    ap.add_argument("--int8-tolerance", type=float, default=0.02,
                    help="fail if quantization moves a score by more than this")
    args = ap.parse_args()

    wanted = args.only or ["embedding", "nli", "minicheck"]
    args.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in wanted:
        print(f"exporting {name}", flush=True)
        if name == "embedding":
            report[name] = export_embedding(args.out)
        else:
            report[name] = export_cross_encoder(name, CROSS_ENCODERS[name], args.out)
        diff = report[name]["max_abs_diff_vs_torch"]
        print(f"  {name}: {report[name]['bytes'] / 1e6:.0f} MB, "
              f"max difference from PyTorch {diff:.2e}", flush=True)

    (args.out / "export-report.json").write_text(json.dumps(report, indent=2) + "\n")
    # Full precision must match PyTorch almost exactly; int8 is allowed to move, but only
    # by less than the recall tolerance the benchmark reports against.
    bad: dict[str, float] = {}
    for name, meta in report.items():
        reference = meta.get("fp32")
        if isinstance(reference, dict):
            diff = float(reference["max_abs_diff_vs_torch"])
            if diff > args.tolerance:
                bad[f"{name}-fp32"] = diff
        diff = float(meta["max_abs_diff_vs_torch"])  # type: ignore[arg-type]
        limit = args.int8_tolerance if meta.get("precision") == "int8" else args.tolerance
        if diff > limit:
            bad[name] = diff
    if bad:
        raise SystemExit(f"exports moved more than allowed: {bad}")
    print(f"wrote {args.out / 'export-report.json'}")


if __name__ == "__main__":
    main()
