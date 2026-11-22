"""The benchmark's own arithmetic: calibrate here, deploy there, rank by honesty."""

from __future__ import annotations

import numpy as np
import pandas as pd

from shiftbench.analysis import budgets, drift, drift_predicts_loss, evaluate_pair, leaderboard, matrix

RNG = np.random.default_rng


def domain(name: str, n: int = 800, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """A domain where unsupported claims score lower, by a margin ``shift`` erodes.

    The evidence is several words of varying length, not one long token. It used to be one
    token, which left every claim in the same length bin and so exercised the mondrian
    threshold on a taxonomy with a single group: the method could not differ from split in
    any test, which is part of why it took a run over real data to notice that it never did.
    """
    rng = RNG(seed)
    unsupported = rng.random(n) < 0.4
    support = np.where(unsupported, rng.normal(0.3 + shift, 0.15, n), rng.normal(0.7, 0.15, n)).clip(0, 1)
    return pd.DataFrame(
        {
            "claim_id": [f"{name}-{i}" for i in range(n)],
            "domain": name,
            "claim": ["c" * (10 + i % 7) for i in range(n)],
            "evidence": [" ".join(["word"] * (20 + i % 40)) for i in range(n)],
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
    rows = {r.method: r for r in evaluate_pair(domain("a", seed=0), domain("b", shift=0.3, seed=1), "fake")}
    assert rows["split"].achieved < rows["split"].promised - 0.02
    assert rows["oracle"].achieved >= rows["oracle"].promised - 0.03


def test_weighting_recovers_some_of_the_loss_without_target_labels() -> None:
    rows = {r.method: r for r in evaluate_pair(domain("a", seed=0), domain("b", shift=0.25, seed=1), "fake")}
    assert rows["weighted"].achieved > rows["split"].achieved


def test_the_matrix_covers_every_ordered_pair_including_each_domain_on_itself() -> None:
    scored = pd.concat([domain("a", seed=0), domain("b", shift=0.25, seed=1)], ignore_index=True)
    pairs = matrix(scored)
    assert set(zip(pairs["source"], pairs["target"], strict=True)) == {
        ("a", "a"),
        ("a", "b"),
        ("b", "a"),
        ("b", "b"),
    }
    # A domain on itself is measured on a held-out half, so it is still honest.
    on_itself = pairs[(pairs["source"] == "a") & (pairs["target"] == "a") & (pairs["method"] == "split")]
    assert float(on_itself["achieved"].iloc[0]) > 0.80


def test_the_leaderboard_breaks_a_tie_on_cost_not_on_over_flagging() -> None:
    """Two detectors keeping every promise are separated by what keeping it cost.

    The tie-break was mean_gap descending, which puts the detector that over-recalls the
    most at the top. Over-recalling is not honesty: the gap is good near zero, not large,
    so it cannot be a one-directional sort key. Built here rather than generated, because
    the matrix holds both directions of every pair and a detector that over-recalls one way
    breaks its promise the other, so the data generator cancels this out.
    """
    pairs = pd.DataFrame(
        [
            {
                "detector": d,
                "source": "x",
                "target": "y",
                "method": "split",
                "threshold": 0.5,
                "promised": 0.90,
                "achieved": achieved,
                "achieved_low": achieved - 0.01,
                "achieved_high": achieved + 0.01,
                "false_alarms": alarms,
                "gap": achieved - 0.90,
                "n_source": 400,
                "n_target": 400,
                "weight_spread": float("nan"),
                "domain_separability": float("nan"),
            }
            for d, achieved, alarms in (
                ("floods-it", 1.00, 0.80),  # keeps the promise by flagging almost everything
                ("precise", 0.91, 0.12),  # keeps it and stays quiet
            )
        ]
    )
    board = leaderboard(pairs, pd.DataFrame(columns=["detector", "labels", "within_tolerance"]))
    assert list(board["promise_kept"]) == [1.0, 1.0]  # the tie is real
    assert next(iter(board["detector"])) == "precise"


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


def test_drift_predicts_the_size_of_the_miss_where_the_promise_breaks() -> None:
    """A label-free warning, measured where it would be acted on.

    This asserted a correlation of 0.3 against the signed loss over every ordered pair and
    got 0.19, which is not the code being wrong. A score-distribution statistic is close to
    direction-blind, so it is nearly equal for a pair and for its reverse, while the loss is
    not: easy source to hard target loses recall, the reverse over-recalls and pins the loss
    at its floor no matter how large the drift. Correlating against the signed loss over
    both directions mixes a relationship with its mirror image.

    Among the pairs where the promise actually breaks, PSI tracks the size of the break
    closely, which is the claim worth making and the one a team could use.
    """
    frames = [domain("a", seed=0)]
    for i, shift in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
        frames.append(domain(f"d{i}", shift=shift, seed=10 + i))
    scored = pd.concat(frames, ignore_index=True)
    pairs = matrix(scored)
    summary = drift_predicts_loss(pairs, drift(scored))
    assert summary["pairs"] == 20
    assert summary["undercovering_pairs"] == 10
    assert summary["psi_spearman_undercovering"] > 0.7
    assert summary["psi_spearman_absolute"] > 0.4
    # The signed figure is the weak one, and it stays reported rather than dropped.
    assert summary["psi_spearman"] < summary["psi_spearman_undercovering"]


def test_a_drift_statistic_cannot_tell_which_way_the_shift_hurts() -> None:
    """Why the signed correlation is weak: the statistic is near enough symmetric.

    PSI bins on the source, so it is not exactly symmetric, but a pair and its reverse come
    out close together while their losses have opposite signs. This is the limitation that
    stops a high reading from being read as "you are under-covering".
    """
    scored = pd.concat([domain("easy", seed=0), domain("hard", shift=0.4, seed=11)], ignore_index=True)
    drifts = drift(scored).set_index(["source", "target"])
    forward = float(drifts.loc[("easy", "hard"), "psi"])
    backward = float(drifts.loc[("hard", "easy"), "psi"])
    assert forward > 0.1
    assert backward > 0.1
    assert abs(forward - backward) < max(forward, backward)

    split = matrix(scored)
    split = split[(split["method"] == "split") & (split["source"] != split["target"])]
    losses = {(r.source, r.target): r.promised - r.achieved for r in split.itertuples()}
    assert losses[("easy", "hard")] > 0.05  # the promise breaks
    assert losses[("hard", "easy")] < 0.0  # the reverse over-recalls


def test_the_leaderboard_ranks_the_honest_detector_first() -> None:
    honest = domain("a", seed=0).assign(detector="honest")
    honest_target = domain("b", seed=1).assign(detector="honest")  # no shift: promise holds
    fragile = domain("a", seed=0).assign(detector="fragile")
    fragile_target = domain("b", shift=0.4, seed=1).assign(detector="fragile")
    scored = pd.concat([honest, honest_target, fragile, fragile_target], ignore_index=True)
    pairs = matrix(scored)
    board = leaderboard(pairs, budgets(scored))
    assert next(iter(board["detector"])) == "honest"
    assert board.set_index("detector").loc["fragile", "worst_gap"] < 0


def test_the_taxonomy_labels_both_sides_with_the_sources_edges() -> None:
    """A mondrian group has to mean the same thing in the target as in the source."""
    from shiftbench.analysis import taxonomy

    source, target = domain("a", seed=0), domain("b", seed=1)
    s_groups, t_groups = taxonomy(source, target)
    assert len(s_groups) == len(source)
    assert len(t_groups) == len(target)
    assert set(t_groups) <= set(s_groups)
    assert len(set(s_groups)) > 1


def test_mondrian_covers_every_target_claim_where_document_groups_covered_none() -> None:
    """The method used group_id, and no target domain holds one of the source's documents.

    Every claim therefore took the pooled fallback, which is the split threshold, so the row
    was split under another name. The taxonomy is defined on both sides, so coverage is full.
    """
    from shiftbench.analysis import taxonomy
    from shiftbench.calibration import mondrian_coverage, mondrian_thresholds

    source, target = domain("a", seed=0), domain("b", shift=0.25, seed=1)
    scores = source["support"].to_numpy()
    unsupported = source["unsupported"].to_numpy()

    documents = np.array([f"a-doc-{i}" for i in range(len(source))])
    their_documents = np.array([f"b-doc-{i}" for i in range(len(target))])
    by_document = mondrian_thresholds(scores, unsupported, documents, 0.9)
    assert mondrian_coverage(by_document, their_documents) == 0.0

    s_groups, t_groups = taxonomy(source, target)
    by_length = mondrian_thresholds(scores, unsupported, s_groups, 0.9)
    assert mondrian_coverage(by_length, t_groups) == 1.0


def test_mondrian_reports_an_interval_like_every_other_method() -> None:
    """It reported nan because it computed its recall by hand instead of through the same call."""
    rows = {r.method: r for r in evaluate_pair(domain("a", seed=0), domain("b", shift=0.25, seed=1), "fake")}
    mondrian = rows["mondrian"]
    assert not np.isnan(mondrian.achieved_low)
    assert mondrian.achieved_low <= mondrian.achieved <= mondrian.achieved_high
    assert mondrian.group_coverage == 1.0
