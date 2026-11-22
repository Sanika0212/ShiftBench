"""The promise and its repair, tested on data whose answer is known."""

from __future__ import annotations

import numpy as np
import pytest

from shiftbench.calibration import (
    achieved_recall,
    apply_mondrian,
    bootstrap_recall,
    corrected_level,
    density_ratio,
    false_alarm_rate,
    labels_to_recover,
    mmd,
    mondrian_thresholds,
    psi,
    reweight,
    threshold_for_recall,
    weighted_quantile,
    weighted_threshold,
)


def population(n: int = 4000, shift: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Support scores for a domain. Unsupported claims score lower, by a margin that
    ``shift`` erodes: a shifted domain is one where the detector separates less well."""
    rng = np.random.default_rng(seed)
    unsupported = rng.random(n) < 0.4
    scores = np.where(unsupported, rng.normal(0.3 + shift, 0.15, n), rng.normal(0.7, 0.15, n))
    return np.clip(scores, 0, 1), unsupported


def test_threshold_keeps_its_promise_on_the_domain_it_was_fitted_on() -> None:
    scores, unsupported = population()
    for target in (0.80, 0.90, 0.95):
        threshold = threshold_for_recall(scores[:2000], unsupported[:2000], target)
        achieved = achieved_recall(scores[2000:], unsupported[2000:], threshold)
        assert abs(achieved - target) < 0.04


def test_a_promise_costs_false_alarms() -> None:
    scores, unsupported = population()
    low = threshold_for_recall(scores, unsupported, 0.80)
    high = threshold_for_recall(scores, unsupported, 0.95)
    assert high > low
    assert false_alarm_rate(scores, unsupported, high) > false_alarm_rate(scores, unsupported, low)


def test_the_promise_breaks_when_the_domain_shifts() -> None:
    """The whole benchmark in one test: fit here, deploy there, lose recall."""
    source_scores, source_unsupported = population(seed=0)
    target_scores, target_unsupported = population(shift=0.25, seed=1)
    threshold = threshold_for_recall(source_scores, source_unsupported, 0.90)
    achieved = achieved_recall(target_scores, target_unsupported, threshold)
    assert achieved < 0.85
    oracle = threshold_for_recall(target_scores, target_unsupported, 0.90)
    assert oracle > threshold  # the target needed a more cautious threshold


def test_bootstrap_interval_covers_the_point_and_narrows_with_data() -> None:
    scores, unsupported = population(n=4000)
    threshold = threshold_for_recall(scores, unsupported, 0.90)
    wide = bootstrap_recall(scores[:200], unsupported[:200], threshold, draws=500)
    narrow = bootstrap_recall(scores, unsupported, threshold, draws=500)
    assert wide.covers(wide.point)
    assert (narrow.high - narrow.low) < (wide.high - wide.low)


def test_weighted_quantile_matches_the_plain_one_when_weights_are_equal() -> None:
    values = np.linspace(0, 1, 101)
    weights = np.ones_like(values)
    assert weighted_quantile(values, weights, 0.9) == pytest.approx(0.9, abs=0.02)


def test_every_scheme_is_calibrated_at_the_same_level() -> None:
    """The weighted scheme was held to a different level than the other three.

    It took the quantile at the promised target while split, mondrian and oracle took it at
    the corrected one, so the weighted threshold was systematically lower and recalled less
    for a reason that has nothing to do with reweighting. With uniform weights the density
    ratio does nothing, so the two must agree exactly. Measured before the fix: 4.0 points
    of recall at 50 unsupported claims, 1.9 at 100, against a 2 point tolerance.
    """
    rng = np.random.default_rng(0)
    for n in (50, 100, 300, 1000):
        scores = rng.normal(size=n)
        unsupported = np.ones(n, dtype=bool)
        plain = threshold_for_recall(scores, unsupported, 0.90)
        weighted = weighted_threshold(scores, unsupported, np.ones(n), 0.90)
        assert weighted == plain, f"schemes disagree at n={n}"


def test_the_threshold_is_the_conformal_order_statistic() -> None:
    """The ceil((n + 1) * target)-th smallest unsupported score, exactly.

    numpy's "higher" method spaces n values over n - 1 intervals, which landed one position
    further up and made the split scheme over-promise: 0.9215 achieved against 0.90 asked
    at 50 claims. It also could not agree with the weighted path, which spaces over n.
    """
    scores = np.arange(1.0, 51.0)  # 50 values, so the order statistic is readable
    unsupported = np.ones(50, dtype=bool)
    expected = float(np.ceil(51 * 0.90))  # the 46th smallest, which is the value 46.0
    assert threshold_for_recall(scores, unsupported, 0.90) == expected
    assert corrected_level(50, 0.90) == pytest.approx(0.92)


def test_density_ratio_is_larger_for_source_claims_that_look_like_the_target() -> None:
    rng = np.random.default_rng(0)
    source = rng.normal(0, 1, (800, 3))
    target = rng.normal(1.5, 1, (800, 3))
    weights = density_ratio(source, target)
    assert weights.shape == (800,)
    # Source rows closest to the target's centre should carry the most weight.
    distance = np.linalg.norm(source - 1.5, axis=1)
    assert np.corrcoef(weights, -distance)[0, 1] > 0.3


def test_weighting_moves_the_threshold_towards_the_target() -> None:
    source_scores, source_unsupported = population(seed=0)
    target_scores, target_unsupported = population(shift=0.25, seed=1)
    plain = threshold_for_recall(source_scores, source_unsupported, 0.90)
    features_source = np.column_stack([source_scores, np.ones_like(source_scores)])
    features_target = np.column_stack([target_scores, np.ones_like(target_scores)])
    weights = density_ratio(features_source, features_target)
    weighted = weighted_threshold(source_scores, source_unsupported, weights, 0.90)
    plain_recall = achieved_recall(target_scores, target_unsupported, plain)
    weighted_recall = achieved_recall(target_scores, target_unsupported, weighted)
    assert weighted_recall > plain_recall


def test_reweighting_reports_when_it_cannot_do_anything() -> None:
    """Separable domains make importance weights an extrapolation, and the output says so.

    The domain classifier saturates when the two domains do not overlap, so every source
    claim gets almost the same weight and the weighted threshold becomes the unweighted one.
    Measured while adding this: the threshold moved 0.096 between overlapping domains and
    0.017 between very different ones, the wrong way round for a shift repair. Effective
    sample size is no use as a warning here, rising from 66% to 94% of the sample as the
    repair stopped working, because near-equal weights look healthy by that measure.
    """
    rng = np.random.default_rng(0)
    overlapping = rng.normal(size=(400, 3))
    nearby = rng.normal(0.3, 1.0, size=(400, 3))
    far = np.column_stack([rng.normal(40, 1.0, 400), rng.normal(900, 5, 400), rng.normal(50, 1, 400)])

    close = reweight(overlapping, nearby, seed=0)
    apart = reweight(overlapping, far, seed=0)

    assert close.separability < 0.95
    assert apart.separability > 0.999
    assert apart.spread < close.spread
    assert apart.degenerate
    assert not close.degenerate


def test_mondrian_uses_a_group_threshold_where_the_group_is_big_enough() -> None:
    scores, unsupported = population(n=2000)
    groups = np.where(np.arange(2000) % 2 == 0, "qa", "summary")
    thresholds = mondrian_thresholds(scores, unsupported, groups, 0.90)
    assert set(thresholds) == {"__pooled__", "qa", "summary"}
    applied = apply_mondrian(thresholds, groups)
    assert applied.shape == (2000,)
    tiny = mondrian_thresholds(scores[:20], unsupported[:20], groups[:20], 0.90, minimum=25)
    assert set(tiny) == {"__pooled__"}  # too small to trust its own quantile


def test_labels_buy_the_promise_back() -> None:
    scores, unsupported = population(shift=0.25, seed=1)
    curve = labels_to_recover(scores, unsupported, 0.90, budgets=(25, 200), draws=20)
    assert set(curve) == {25, 200}
    assert curve[200]["within_tolerance"] >= curve[25]["within_tolerance"]
    assert curve[200]["recall_sd"] < curve[25]["recall_sd"]


def test_drift_statistics_see_the_shift() -> None:
    source, _ = population(seed=0)
    same, _ = population(seed=2)
    shifted, _ = population(shift=0.3, seed=3)
    assert psi(source, shifted) > psi(source, same)
    _, p_same = mmd(source, same, permutations=100)
    statistic, p_shift = mmd(source, shifted, permutations=100)
    assert p_shift < 0.05
    assert p_same > 0.01
    assert statistic > 0


def test_calibrating_without_an_unsupported_claim_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        threshold_for_recall(np.linspace(0, 1, 10), np.zeros(10, dtype=bool), 0.9)
