"""The benchmark's own arithmetic: calibrate here, deploy there, rank by honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shiftbench.analysis import budgets, drift, drift_predicts_loss, evaluate_pair, leaderboard, matrix

RNG = np.random.default_rng


def domain(name: str, n: int = 800, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A domain where unsupported claims score lower, by a margin ``shift`` erodes."""
    rng = RNG(seed)
    unsupported = rng.random(n) < 0.4
    support = np.where(
        unsupported, rng.normal(0.3 + shift, 0.15, n), rng.normal(0.7, 0.15, n)
    ).clip(0, 1)
    return pd.DataFrame(
        {
            "claim_id": [f"{name}-{i}" for i in range(n)],
            "domain": name,
            "claim": ["c" * (10 + i % 7) for i in range(n)],
            "evidence": ["e" * (40 + i % 11) for i in range(n)],
            "supported": ~unsupported,
            "unsupported": unsupported,
            "group_id": [f"g{i % 4}" for i in range(n)],
            "source": name,
            "support": support,
            "detector": "fake",
        }
    )


def test_every_method_is_measured_for_a_pair() -> None:
    rows = evaluate_pair(domain("a", seed=0), domain("b", shift=0.25, seed=1), "fake")
    methods = {r.method for r in rows}
    assert methods == {"split", "weighted", "oracle", "mondrian"}
    for r in rows:
        assert r.n_source == 800
        assert 0.0 <= r.achieved <= 1.0
        assert r.gap == r.achieved - r.promised


def test_the_shipped_threshold_loses_recall_under_shift_and_the_oracle_does_not() -> None:
    """The whole benchmark in one assertion."""
    rows = {r.method: r for r in
            evaluate_pair(domain("a", seed=0), domain("b", shift=0.3, seed=1), "fake")}
    assert rows["split"].achieved < rows["split"].promised - 0.02
    assert rows["oracle"].achieved >= rows["oracle"].promised - 0.03


def test_weighting_recovers_some_of_the_loss_without_target_labels() -> None:
    rows = {r.method: r for r in
            evaluate_pair(domain("a", seed=0), domain("b", shift=0.25, seed=1), "fake")}
    assert rows["weighted"].achieved > rows["split"].achieved


def test_the_matrix_covers_every_ordered_pair_including_each_domain_on_itself() -> None:
    scored = pd.concat([domain("a", seed=0), domain("b", shift=0.25, seed=1)], ignore_index=True)
    pairs = matrix(scored)
    assert set(zip(pairs["source"], pairs["target"], strict=True)) == {
        ("a", "a"), ("a", "b"), ("b", "a"), ("b", "b")
    }
    # A domain on itself is measured on a held-out half, so it is still honest.
    on_itself = pairs[(pairs["source"] == "a") & (pairs["target"] == "a") &
                      (pairs["method"] == "split")]
    assert float(on_itself["achieved"].iloc[0]) > 0.80


def test_more_labels_buy_the_promise_back() -> None:
    curves = budgets(pd.concat([domain("b", shift=0.25, seed=1)], ignore_index=True))
    biggest = curves.sort_values("labels").iloc[-1]
    smallest = curves.sort_values("labels").iloc[0]
    assert biggest["within_tolerance"] >= smallest["within_tolerance"]
    assert biggest["recall_sd"] < smallest["recall_sd"]


def test_drift_statistics_see_the_shift_and_not_the_absence_of_one() -> None:
    scored = pd.concat(
        [domain("a", seed=0), domain("same", seed=2), domain("shifted", shift=0.3, seed=3)],
        ignore_index=True,
    )
    drifts = drift(scored).set_index(["source", "target"])
    assert drifts.loc[("a", "shifted"), "psi"] > drifts.loc[("a", "same"), "psi"]
    assert drifts.loc[("a", "shifted"), "mmd_p_value"] < 0.05


def test_drift_correlates_with_the_recall_actually_lost() -> None:
    frames = [domain("a", seed=0)]
    for i, shift in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
        frames.append(domain(f"d{i}", shift=shift, seed=10 + i))
    scored = pd.concat(frames, ignore_index=True)
    pairs = matrix(scored)
    summary = drift_predicts_loss(pairs, drift(scored))
    assert summary["pairs"] >= 4
    assert summary["psi_spearman"] > 0.3  # a label-free warning worth acting on


def test_the_leaderboard_ranks_the_honest_detector_first() -> None:
    honest = domain("a", seed=0).assign(detector="honest")
    honest_target = domain("b", seed=1).assign(detector="honest")  # no shift: promise holds
    fragile = domain("a", seed=0).assign(detector="fragile")
    fragile_target = domain("b", shift=0.4, seed=1).assign(detector="fragile")
    scored = pd.concat([honest, honest_target, fragile, fragile_target], ignore_index=True)
    pairs = matrix(scored)
    board = leaderboard(pairs, budgets(scored))
    assert list(board["detector"])[0] == "honest"
    assert board.set_index("detector").loc["fragile", "worst_gap"] < 0
