"""The leaderboard page: it renders before there are results, and reads them when there are."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shiftbench.site import load_results, render


def board() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"detector": "minicheck", "pairs": 12, "mean_gap": -0.01, "worst_gap": -0.04,
             "promise_kept": 0.75, "mean_false_alarms": 0.21, "labels_to_recover": 100.0},
            {"detector": "word-overlap", "pairs": 12, "mean_gap": -0.18, "worst_gap": -0.41,
             "promise_kept": 0.08, "mean_false_alarms": 0.55, "labels_to_recover": float("nan")},
        ]
    )


def pairs() -> pd.DataFrame:
    rows = []
    for source in ("legal", "medical"):
        for target in ("legal", "medical"):
            rows.append({"detector": "minicheck", "source": source, "target": target,
                         "method": "split", "achieved": 0.91 if source == target else 0.72})
    return pd.DataFrame(rows)


def test_the_page_says_what_to_do_when_there_are_no_results() -> None:
    html = render({})
    assert "shiftbench analyse" in html
    assert "<title>ShiftBench</title>" in html


def test_the_page_lists_every_detector_and_its_numbers() -> None:
    html = render({"leaderboard": board()})
    assert "minicheck" in html
    assert "word-overlap" in html
    assert "75%" in html  # promise kept, formatted
    assert "100" in html  # labels to recover


def test_a_detector_that_never_recovers_shows_a_dash_not_a_nan() -> None:
    html = render({"leaderboard": board()})
    assert "nan" not in html.lower()
    assert "n/a" in html


def test_the_matrix_marks_a_kept_promise_differently_from_a_broken_one() -> None:
    html = render({"leaderboard": board(), "pairs": pairs()}, promised=0.90)
    assert 'class="cell good"' in html  # the diagonal, where the threshold was fitted
    assert 'class="cell bad"' in html  # off diagonal, where it was not
    assert "calibrated on" in html


def test_the_page_declares_both_themes() -> None:
    """A page read in a dark editor must not be a white rectangle."""
    html = render({"leaderboard": board()})
    assert "prefers-color-scheme: dark" in html
    assert "viewport" in html


def test_results_load_only_the_tables_that_exist(tmp_path: Path) -> None:
    board().to_parquet(tmp_path / "leaderboard.parquet", index=False)
    loaded = load_results(tmp_path)
    assert set(loaded) == {"leaderboard"}
    assert len(loaded["leaderboard"]) == 2
