"""Scoring is resumable, and an incomplete cache is never silently analysed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shiftbench.domains import COLUMNS
from shiftbench.runner import (
    cache_path,
    chunks,
    features,
    import_scores,
    read_cache,
    score_domain,
    scored_frame,
)


class Counting:
    """A detector that records how many claims it was asked to score."""

    name = "counting"

    def __init__(self) -> None:
        self.seen: list[str] = []

    def score(self, claims: list[str], evidence: list[str]) -> np.ndarray:
        self.seen += claims
        return np.linspace(0.1, 0.9, len(claims)).astype(np.float32)


def claims(n: int = 10, domain: str = "legal") -> pd.DataFrame:
    rows = [
        {
            "claim_id": f"{domain}-{i}",
            "domain": domain,
            "claim": f"claim {i}",
            "evidence": f"evidence {i}",
            "supported": i % 3 != 0,
            "group_id": f"doc-{i // 3}",
            "source": domain,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def test_chunks_cover_every_row_once() -> None:
    frame = claims(10)
    seen = pd.concat(list(chunks(frame, 4)))
    assert len(seen) == 10
    assert sorted(seen["claim_id"]) == sorted(frame["claim_id"])


def test_scoring_writes_a_cache_and_resumes_from_it(tmp_path: Path) -> None:
    frame = claims(8)
    first = Counting()
    out = score_domain(frame, "counting", first, chunk=4, duty=1.0, root=tmp_path, headroom=0.0)
    assert len(out) == 8
    assert out["support"].notna().all()
    assert len(first.seen) == 8

    # Second pass: everything is cached, so the detector is never asked again.
    second = Counting()
    again = score_domain(frame, "counting", second, chunk=4, duty=1.0, root=tmp_path, headroom=0.0)
    assert second.seen == []
    assert np.allclose(again["support"].to_numpy(), out["support"].to_numpy())


def test_scoring_only_asks_for_what_is_missing(tmp_path: Path) -> None:
    frame = claims(8)
    score_domain(frame.head(4), "counting", Counting(), chunk=4, duty=1.0, root=tmp_path,
                 headroom=0.0)
    detector = Counting()
    score_domain(frame, "counting", detector, chunk=4, duty=1.0, root=tmp_path, headroom=0.0)
    assert len(detector.seen) == 4  # the four that were not in the cache


def test_imported_scores_land_in_the_same_cache(tmp_path: Path) -> None:
    rows = [{"claim_id": "legal-0", "support": 0.2}, {"claim_id": "legal-1", "support": 0.8}]
    assert import_scores(rows, "legal", "minicheck", root=tmp_path) == 2
    assert import_scores(rows, "legal", "minicheck", root=tmp_path) == 0  # idempotent
    cached = read_cache(cache_path("legal", "minicheck", tmp_path))
    assert cached == {"legal-0": 0.2, "legal-1": 0.8}


def test_an_incomplete_cache_is_skipped_not_analysed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Half a domain's scores would silently bias every number downstream."""
    frame = claims(8)
    score_domain(frame.head(4), "counting", Counting(), chunk=4, duty=1.0, root=tmp_path,
                 headroom=0.0)
    with pytest.raises(RuntimeError, match="no detector has a complete set"):
        scored_frame(frame, ["counting"], root=tmp_path)
    assert "not scored yet" in capsys.readouterr().out


def test_scored_frame_returns_the_detectors_that_are_complete(tmp_path: Path) -> None:
    frame = claims(8)
    score_domain(frame, "counting", Counting(), chunk=4, duty=1.0, root=tmp_path, headroom=0.0)
    out = scored_frame(frame, ["counting"], root=tmp_path)
    assert set(out["detector"]) == {"counting"}
    assert len(out) == 8


def test_features_are_the_score_and_two_lengths() -> None:
    frame = claims(5)
    frame["support"] = np.linspace(0, 1, 5)
    f = features(frame)
    assert f.shape == (5, 3)
    assert np.allclose(f[:, 0], frame["support"].to_numpy())
    assert np.allclose(f[:, 1], frame["claim"].str.len().to_numpy())
