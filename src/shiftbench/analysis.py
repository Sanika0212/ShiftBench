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
    achieved_recall,
    apply_mondrian,
    bootstrap_recall,
    density_ratio,
    false_alarm_rate,
    labels_to_recover,
    mmd,
    mondrian_thresholds,
    psi,
    threshold_for_recall,
    weighted_threshold,
)
from shiftbench.runner import features

TARGET = 0.90
TOLERANCE = 0.02


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

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def _split(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return frame["support"].to_numpy(dtype=float), frame["unsupported"].to_numpy(dtype=bool)


def evaluate_pair(source: pd.DataFrame, target: pd.DataFrame, detector: str,
                  promised: float = TARGET, seed: int = 0) -> list[Pair]:
    """Every threshold method, calibrated on ``source`` and measured on ``target``."""
    s_scores, s_unsupported = _split(source)
    t_scores, t_unsupported = _split(target)
    rows: list[Pair] = []

    def add(method: str, threshold: float) -> None:
        interval = bootstrap_recall(t_scores, t_unsupported, threshold, seed=seed)
        rows.append(
            Pair(
                detector=detector,
                source=str(source["domain"].iloc[0]),
                target=str(target["domain"].iloc[0]),
                method=method,
                threshold=float(threshold),
                promised=promised,
                achieved=interval.point,
                achieved_low=interval.low,
                achieved_high=interval.high,
                false_alarms=false_alarm_rate(t_scores, t_unsupported, threshold),
                gap=interval.point - promised,
                n_source=len(source),
                n_target=len(target),
            )
        )

    add("split", threshold_for_recall(s_scores, s_unsupported, promised))
    weights = density_ratio(features(source), features(target), seed=seed)
    add("weighted", weighted_threshold(s_scores, s_unsupported, weights, promised))
    add("oracle", threshold_for_recall(t_scores, t_unsupported, promised))

    if "group_id" in source and source["group_id"].notna().any():
        thresholds = mondrian_thresholds(
            s_scores, s_unsupported, source["group_id"].to_numpy(), promised
        )
        per_claim = apply_mondrian(thresholds, target["group_id"].to_numpy())
        recall = float(np.mean(t_scores[t_unsupported] <= per_claim[t_unsupported]))
        rows.append(
            Pair(
                detector=detector,
                source=str(source["domain"].iloc[0]),
                target=str(target["domain"].iloc[0]),
                method="mondrian",
                threshold=float(np.median(per_claim)),
                promised=promised,
                achieved=recall,
                achieved_low=float("nan"),
                achieved_high=float("nan"),
                false_alarms=float(np.mean(t_scores[~t_unsupported] <= per_claim[~t_unsupported])),
                gap=recall - promised,
                n_source=len(source),
                n_target=len(target),
            )
        )
    return rows


def matrix(scored: pd.DataFrame, promised: float = TARGET, seed: int = 0) -> pd.DataFrame:
    """Every detector by every ordered pair of domains, including each domain on itself."""
    rows: list[dict[str, Any]] = []
    for detector, per_detector in scored.groupby("detector"):
        domains = sorted(per_detector["domain"].unique())
        for source in domains:
            s = per_detector[per_detector["domain"] == source]
            for target in domains:
                t = per_detector[per_detector["domain"] == target]
                if source == target:
                    # Held out half, so a domain on itself is still an honest measurement.
                    rng = np.random.default_rng(seed)
                    half = rng.random(len(s)) < 0.5
                    s, t = s[half], s[~half]
                rows += [p.as_row() for p in
                         evaluate_pair(s, t, str(detector), promised=promised, seed=seed)]
    return pd.DataFrame(rows)


def budgets(scored: pd.DataFrame, promised: float = TARGET, tolerance: float = TOLERANCE,
            seed: int = 0) -> pd.DataFrame:
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

    Spearman correlation across pairs between each statistic and the recall actually lost by
    the shipped (split) threshold.
    """
    from scipy.stats import spearmanr

    split = pairs[(pairs["method"] == "split") & (pairs["source"] != pairs["target"])]
    merged = split.merge(drifts, on=["detector", "source", "target"], how="inner")
    if len(merged) < 4:
        return {"pairs": float(len(merged))}
    loss = (merged["promised"] - merged["achieved"]).to_numpy()
    return {
        "pairs": float(len(merged)),
        "psi_spearman": float(spearmanr(merged["psi"].to_numpy(), loss).statistic),
        "mmd_spearman": float(spearmanr(merged["mmd"].to_numpy(), loss).statistic),
    }


def leaderboard(pairs: pd.DataFrame, budget_curves: pd.DataFrame,
                tolerance: float = TOLERANCE) -> pd.DataFrame:
    """Rank detectors by honesty under shift, not by how well they separate in one domain."""
    shifted = pairs[(pairs["method"] == "split") & (pairs["source"] != pairs["target"])]
    rows = []
    for detector, group in shifted.groupby("detector"):
        kept = group[group["achieved"] >= group["promised"] - tolerance]
        recovered = budget_curves[budget_curves["detector"] == detector]
        enough = recovered[recovered["within_tolerance"] >= 0.9]
        rows.append(
            {
                "detector": detector,
                "pairs": len(group),
                "mean_gap": float(group["gap"].mean()),
                "worst_gap": float(group["gap"].min()),
                "promise_kept": float(len(kept) / len(group)) if len(group) else float("nan"),
                "mean_false_alarms": float(group["false_alarms"].mean()),
                "labels_to_recover": float(enough["labels"].min()) if len(enough) else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["promise_kept", "mean_gap"], ascending=False)
