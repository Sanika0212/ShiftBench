"""The benchmark itself: calibrate in one domain, deploy in another, count the damage.

For every (detector, source domain, target domain) the analysis does what a team would do:
fit a threshold on the source so the detector catches a promised share of unsupported
claims, then apply that threshold to the target and measure what it actually catches.

Four thresholds are compared, because the repair matters as much as the damage:

1. **split** conformal on the source, the thing a team would ship;
2. **weighted** conformal, reweighting source claims towards the target with a density
   ratio, which needs no target labels;
3. **mondrian**, a threshold per group inside the source;
4. **oracle**, fitted on the target's own labels, which is the ceiling nobody has.

Then the label budget: how many target labels it takes to get back within tolerance, and
whether a drift statistic computed without labels would have warned first.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from shiftbench.calibration import (
    Reweighting,
    apply_mondrian,
    bootstrap_recall,
    false_alarm_rate,
    labels_to_recover,
    mmd,
    mondrian_coverage,
    mondrian_thresholds,
    psi,
    reweight,
    threshold_for_recall,
    weighted_threshold,
)
from shiftbench.runner import features

TARGET = 0.90
TOLERANCE = 0.02
MONDRIAN_BINS = 4


@dataclass
class Pair:
    detector: str
    source: str
    target: str
    method: str
    threshold: float
    promised: float
    achieved: float
    achieved_low: float
    achieved_high: float
    false_alarms: float
    gap: float
    n_source: int
    n_target: int
    # Only the weighted method fills these in; nan elsewhere. A weighted row whose spread is
    # near zero did not reweight anything, and its achieved recall is the split one wearing
    # another name. Reported rather than left to be inferred from the two being equal.
    weight_spread: float = float("nan")
    domain_separability: float = float("nan")
    # Only the mondrian method fills this in: the share of target claims whose own group had
    # a threshold rather than the pooled fallback. Zero means the method never ran.
    group_coverage: float = float("nan")

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame["support"].to_numpy(dtype=float), frame["unsupported"].to_numpy(dtype=bool)


def taxonomy(
    fit: pd.DataFrame, measure: pd.DataFrame, bins: int = MONDRIAN_BINS
) -> tuple[np.ndarray, np.ndarray]:
    """A group label that exists on both sides, which is what a Mondrian threshold needs.

    ``group_id`` cannot do this job, though it was what the method used. It exists to keep
    one document's claims together so a split never puts a document on both sides, and it is
    specific to its domain twice over: legal carries 591 groups across 600 claims, so no
    group holds the minimum number of unsupported claims a threshold needs, and no target
    domain contains a single one of the source's groups. Every claim therefore took the
    pooled fallback, and the method was the split threshold wearing another name: identical
    to split, in threshold and in achieved recall, on all 20 cross-domain pairs and on 4 of
    the 5 same-domain ones. Two rows agreeing was the only sign of it.

    Evidence length in words is label-free, defined in every domain, and coarse enough that
    each bin holds enough unsupported claims to fit on. The edges are quantiles of the
    source alone, so nothing about the target enters its own threshold.
    """

    def words(frame: pd.DataFrame) -> np.ndarray:
        return frame["evidence"].str.count(r"\S+").to_numpy(dtype=float)

    source_words, target_words = words(fit), words(measure)
    edges = np.quantile(source_words, np.linspace(0, 1, bins + 1)[1:-1])

    def label(values: np.ndarray) -> np.ndarray:
        return np.array([f"length-{i}" for i in np.digitize(values, edges)])

    return label(source_words), label(target_words)


def evaluate_pair(
    source: pd.DataFrame, target: pd.DataFrame, detector: str, promised: float = TARGET, seed: int = 0
) -> list[Pair]:
    """Every threshold method, calibrated on ``source`` and measured on ``target``."""
    s_scores, s_unsupported = _split(source)
    t_scores, t_unsupported = _split(target)
    rows: list[Pair] = []

    def add(
        method: str,
        threshold: float | np.ndarray,
        reweighting: Reweighting | None = None,
        coverage: float = float("nan"),
    ) -> None:
        interval = bootstrap_recall(t_scores, t_unsupported, threshold, seed=seed)
        rows.append(
            Pair(
                detector=detector,
                source=str(source["domain"].iloc[0]),
                target=str(target["domain"].iloc[0]),
                method=method,
                threshold=float(np.median(threshold)),
                promised=promised,
                achieved=interval.point,
                achieved_low=interval.low,
                achieved_high=interval.high,
                false_alarms=false_alarm_rate(t_scores, t_unsupported, threshold),
                gap=interval.point - promised,
                n_source=len(source),
                n_target=len(target),
                weight_spread=float("nan") if reweighting is None else reweighting.spread,
                domain_separability=(float("nan") if reweighting is None else reweighting.separability),
                group_coverage=coverage,
            )
        )

    add("split", threshold_for_recall(s_scores, s_unsupported, promised))
    reweighting = reweight(features(source), features(target), seed=seed)
    add(
        "weighted",
        weighted_threshold(s_scores, s_unsupported, reweighting.weights, promised),
        reweighting,
    )
    add("oracle", threshold_for_recall(t_scores, t_unsupported, promised))

    s_groups, t_groups = taxonomy(source, target)
    thresholds = mondrian_thresholds(s_scores, s_unsupported, s_groups, promised)
    add(
        "mondrian",
        apply_mondrian(thresholds, t_groups),
        coverage=mondrian_coverage(thresholds, t_groups),
    )
    return rows


def halves(frame: pd.DataFrame, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a domain in two, keeping the share of unsupported claims the same in each.

    A domain measured against itself needs a held-out half, and the half has to be drawn
    inside each class rather than over the rows. An unstratified draw can hand one side
    most of the unsupported claims and the other almost none, and the threshold is fitted
    on unsupported claims alone: on the test fixture here it produced 302 of them on one
    side and none at all on the other, which is not a measurement, it is a crash.

    That happened because the draw used a fixed seed and the fixture's own labels came from
    the same stream, so "below 0.5" was nested inside "below 0.4" and the two masks agreed
    perfectly. Stratifying removes the whole class of accident: neither side can come out
    empty whatever the stream does, so the measurement no longer depends on the data having
    been built with a different seed than the code reading it.
    """
    rng = np.random.default_rng(seed)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    unsupported = frame["unsupported"].to_numpy(dtype=bool)
    for mask in (unsupported, ~unsupported):
        idx = np.flatnonzero(mask)
        rng.shuffle(idx)
        cut = len(idx) // 2
        left.append(idx[:cut])
        right.append(idx[cut:])
    return frame.iloc[np.sort(np.concatenate(left))], frame.iloc[np.sort(np.concatenate(right))]


def matrix(scored: pd.DataFrame, promised: float = TARGET, seed: int = 0) -> pd.DataFrame:
    """Every detector by every ordered pair of domains, including each domain on itself."""
    rows: list[dict[str, Any]] = []
    for detector, per_detector in scored.groupby("detector"):
        domains = sorted(per_detector["domain"].unique())
        for source in domains:
            whole = per_detector[per_detector["domain"] == source]
            for target in domains:
                # Bound to new names rather than reassigned: rebinding the source inside
                # this loop left every pair after the diagonal calibrated on half of it.
                if source == target:
                    fit, measure = halves(whole, seed)
                else:
                    fit = whole
                    measure = per_detector[per_detector["domain"] == target]
                rows += [
                    p.as_row()
                    for p in evaluate_pair(fit, measure, str(detector), promised=promised, seed=seed)
                ]
    return pd.DataFrame(rows)


def budgets(
    scored: pd.DataFrame, promised: float = TARGET, tolerance: float = TOLERANCE, seed: int = 0
) -> pd.DataFrame:
    """How many target labels bring the promise back, per detector and target domain."""
    rows = []
    for (detector, domain), group in scored.groupby(["detector", "domain"]):
        scores, unsupported = _split(group)
        curve = labels_to_recover(scores, unsupported, promised, tolerance=tolerance, seed=seed)
        for k, stats in curve.items():
            rows.append({"detector": detector, "target": domain, "labels": k, **stats})
    return pd.DataFrame(rows)


def drift(scored: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Label-free statistics on the score distributions, per ordered pair of domains."""
    rows = []
    for detector, per_detector in scored.groupby("detector"):
        domains = sorted(per_detector["domain"].unique())
        for source in domains:
            s = per_detector[per_detector["domain"] == source]["support"].to_numpy(dtype=float)
            for target in domains:
                if source == target:
                    continue
                t = per_detector[per_detector["domain"] == target]["support"].to_numpy(dtype=float)
                statistic, p_value = mmd(s, t, seed=seed)
                rows.append(
                    {
                        "detector": detector,
                        "source": source,
                        "target": target,
                        "psi": psi(s, t),
                        "mmd": statistic,
                        "mmd_p_value": p_value,
                    }
                )
    return pd.DataFrame(rows)


def drift_predicts_loss(pairs: pd.DataFrame, drifts: pd.DataFrame) -> dict[str, float]:
    """Would a drift statistic have warned before the labels arrived?

    Spearman correlation across ordered pairs between each statistic and the recall the
    shipped (split) threshold actually lost, reported three ways, because one number here
    is misleading.

    A drift statistic on the scores is close to direction-blind: it is nearly the same for a
    pair and for its reverse, while the loss is not. Shipping a threshold from an easy
    domain to a hard one loses recall; the reverse over-recalls, and that direction pins the
    loss at its floor however large the drift is. So a correlation taken against the signed
    loss over every pair mixes a relationship with its own mirror image and comes out weak
    for a reason that is arithmetic rather than empirical.

    What a team can act on is whether the promise is about to break, and by how much, among
    the pairs where it does break. That is ``*_undercovering``. The correlation against the
    absolute loss is reported too, as the version that does not condition on the answer.
    Measured on the synthetic fixture in the tests: 0.88 undercovering, 0.57 absolute, 0.19
    signed. A high statistic is a reason to recalibrate and measure, not a reading of which
    way the promise moved.
    """
    from scipy.stats import spearmanr

    def rho(x: np.ndarray, y: np.ndarray) -> float:
        if len(x) < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
            return float("nan")
        return float(spearmanr(x, y).statistic)

    split = pairs[(pairs["method"] == "split") & (pairs["source"] != pairs["target"])]
    merged = split.merge(drifts, on=["detector", "source", "target"], how="inner")
    if len(merged) < 4:
        return {"pairs": float(len(merged))}
    loss = (merged["promised"] - merged["achieved"]).to_numpy()
    under = merged[loss > 0]
    under_loss = (under["promised"] - under["achieved"]).to_numpy()
    out = {
        "pairs": float(len(merged)),
        "undercovering_pairs": float(len(under)),
    }
    for name in ("psi", "mmd"):
        statistic = merged[name].to_numpy()
        out[f"{name}_spearman"] = rho(statistic, loss)
        out[f"{name}_spearman_absolute"] = rho(statistic, np.abs(loss))
        out[f"{name}_spearman_undercovering"] = rho(under[name].to_numpy(), under_loss)
    return out


def leaderboard(
    pairs: pd.DataFrame, budget_curves: pd.DataFrame, tolerance: float = TOLERANCE
) -> pd.DataFrame:
    """Rank detectors by honesty under shift, not by how well they separate in one domain."""
    shifted = pairs[(pairs["method"] == "split") & (pairs["source"] != pairs["target"])]
    rows = []
    for detector, group in shifted.groupby("detector"):
        kept = group[group["achieved"] >= group["promised"] - tolerance]
        recovered = budget_curves[budget_curves["detector"] == detector]
        # The bar has to clear on every target, not on the easiest one. Taking the minimum
        # budget over all (detector, target) rows published the best case as the case: it
        # reported 50 labels for word-overlap because one target of five crossed 0.9 at that
        # budget while the other four sat between 0.78 and 0.84, and the same detector had no
        # target above the bar at 100 or 200. A budget that more labels undo is not a budget.
        floor = recovered.groupby("labels")["within_tolerance"].min()
        enough = floor[floor >= 0.9]
        rows.append(
            {
                "detector": detector,
                "pairs": len(group),
                "mean_gap": float(group["gap"].mean()),
                "worst_gap": float(group["gap"].min()),
                "promise_kept": float(len(kept) / len(group)) if len(group) else float("nan"),
                "mean_false_alarms": float(group["false_alarms"].mean()),
                "labels_to_recover": float(enough.index.min()) if len(enough) else float("nan"),
                # How close the hardest target got, at the budget that suited it best. Without
                # it an empty labels_to_recover cannot be told from one that was never measured.
                "hardest_target_recovery": float(floor.max()) if len(floor) else float("nan"),
            }
        )
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    # Most promises kept first, and among detectors that keep as many, the one that keeps
    # them for the fewest false alarms. The tie-break used to be mean_gap descending, which
    # ranks the detector that over-recalls the most, and over-recalling is not honesty: it
    # is flagging more than asked and charging the reader for it. A gap is good when it is
    # near zero, not when it is large, so it cannot be a one-directional sort key.
    return board.sort_values(["promise_kept", "mean_false_alarms"], ascending=[False, True]).reset_index(
        drop=True
    )
