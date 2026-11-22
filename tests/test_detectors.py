"""The detectors, and which of their outputs means supported."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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
    with pytest.raises(ValueError):
        WordOverlap().score(["one", "two"], ["only one passage"])

def test_the_supported_class_comes_from_the_model_config(tmp_path: Path) -> None:
    """A fixed -1 read neutral as support on the NLI model this benchmark uses.

    cross-encoder/nli-deberta-v3-base labels 0 contradiction, 1 entailment, 2 neutral, so
    the last output is neutral. The scientific and medical scores are imported from a
    sibling project that picks entailment by name, so the same detector would have meant
    two different things across the domains being compared.
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
