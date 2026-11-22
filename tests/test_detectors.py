"""The detectors, and which of their outputs means supported."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shiftbench.detectors import WordOverlap, content_words


def test_word_overlap_scores_what_the_evidence_mentions() -> None:
    detector = WordOverlap()
    scores = detector.score(
        ["aspirin reduces fever", "aspirin reduces fever"],
        ["aspirin is known to reduce fever in adults", "the weather today is pleasant"],
    )
    assert scores[0] > scores[1]
    assert 0.0 <= scores[1] <= 1.0


def test_word_overlap_scores_a_claim_with_no_content_words_as_supported() -> None:
    """Nothing to check means nothing to flag, rather than a divide by zero."""
    assert WordOverlap().score(["the and of"], ["anything at all"])[0] == 1.0


def test_content_words_drops_stopwords_and_short_tokens() -> None:
    assert content_words("The cat is on a mat") == {"cat", "mat"}


def test_word_overlap_refuses_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="shorter than"):
        WordOverlap().score(["one", "two"], ["only one passage"])


def test_the_supported_class_comes_from_the_model_config(tmp_path: Path) -> None:
    """A fixed -1 read neutral as support on the NLI model this benchmark uses.

    cross-encoder/nli-deberta-v3-base labels 0 contradiction, 1 entailment, 2 neutral, so
    the last output is neutral. A detector name has to mean one computation across every
    domain being compared, and reading support off a fixed position does not guarantee it.
    """
    from shiftbench.detectors import supported_class

    nli = tmp_path / "nli"
    nli.mkdir()
    (nli / "config.json").write_text(
        json.dumps({"id2label": {"0": "contradiction", "1": "entailment", "2": "neutral"}})
    )
    assert supported_class(nli) == 1

    minicheck = tmp_path / "minicheck"
    minicheck.mkdir()
    (minicheck / "config.json").write_text(json.dumps({"id2label": {"0": "0", "1": "1"}}))
    assert supported_class(minicheck) == 1

    # Three unnamed classes: refused rather than guessed, because a guess here is silent.
    odd = tmp_path / "odd"
    odd.mkdir()
    (odd / "config.json").write_text(json.dumps({"id2label": {"0": "a", "1": "b", "2": "c"}}))
    with pytest.raises(ValueError, match="which of"):
        supported_class(odd)


def test_token_limit_stops_at_the_models_position_table(tmp_path: Path) -> None:
    """RoBERTa's 514 rows start at index 2, so 512 is the limit and 513 fails outright.

    This was a hardcoded 1024. It scored the two domains whose pairs happen to be short and
    then raised "invalid expand shape" from onnxruntime on the first long one, which is the
    worse of the two ways to be wrong: every score it did produce came from a context longer
    than the export was traced and parity-checked at.
    """
    from shiftbench.detectors import token_limit

    roberta = tmp_path / "minicheck"
    roberta.mkdir()
    (roberta / "config.json").write_text(
        json.dumps({"model_type": "roberta", "max_position_embeddings": 514})
    )
    (roberta / "meta.json").write_text(json.dumps({"max_length": 512}))
    assert token_limit(roberta) == 512
    # A smaller request wins; a larger one does not.
    assert token_limit(roberta, 256) == 256
    assert token_limit(roberta, 4096) == 512


def test_token_limit_honours_what_the_export_was_traced_at(tmp_path: Path) -> None:
    """meta.json is the lower ceiling for a model whose table would allow more."""
    from shiftbench.detectors import token_limit

    bert = tmp_path / "embedding"
    bert.mkdir()
    (bert / "config.json").write_text(json.dumps({"model_type": "bert", "max_position_embeddings": 512}))
    (bert / "meta.json").write_text(json.dumps({"max_length": 128}))
    assert token_limit(bert) == 128


def test_token_limit_ignores_the_table_for_relative_attention(tmp_path: Path) -> None:
    """DeBERTa has no absolute position table, so only the traced length applies."""
    from shiftbench.detectors import token_limit

    deberta = tmp_path / "nli"
    deberta.mkdir()
    (deberta / "config.json").write_text(
        json.dumps({"model_type": "deberta-v2", "max_position_embeddings": 512, "relative_attention": True})
    )
    (deberta / "meta.json").write_text(json.dumps({"max_length": 768}))
    assert token_limit(deberta) == 768
