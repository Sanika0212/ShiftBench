"""Promising a recall, and finding out what the promise is worth in another domain.

A deployed grounding checker is a score and a threshold. The threshold is chosen on labelled
data so that the checker catches a promised share of unsupported claims, say 90%. This
module is about what that promise is actually worth when the claims come from somewhere
else, and what it costs to get it back.

1. ``threshold_for_recall`` picks the threshold on a source domain, with the finite-sample
   correction that makes the promise hold on exchangeable data rather than on average.
2. ``achieved_recall`` measures what the same threshold catches on a target domain, with a
   bootstrap interval, because a gap of two points on 300 claims is not a gap.
3. ``weighted_threshold`` reweights the source claims by how much they look like the target,
   using a density ratio from a domain classifier. This is the standard repair for covariate
   shift, and it needs no target labels.
4. ``mondrian_threshold`` calibrates per group (per task, per answer length band), which
   helps when the mixture changes rather than the relationship.
5. ``labels_to_recover`` answers the practical question: how many labelled target claims
   does it take to get the promise back within a tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RNG = np.random.default_rng


def threshold_for_recall(scores: np.ndarray, unsupported: np.ndarray, target: float) -> float:
    """The largest threshold whose recall on these claims is at least ``target``.

    Scores are "how supported this claim looks", so a claim is flagged when its score falls
    below the threshold. With ``n`` unsupported claims the quantile is taken at
    ``ceil((n + 1) * target) / n``, the conformal correction: on a new exchangeable sample
    the promise then holds with the stated probability rather than only in expectation.
    """
    flagged = np.asarray(scores)[np.asarray(unsupported, dtype=bool)]
    if flagged.size == 0:
        raise ValueError("no unsupported claims to calibrate on")
    n = flagged.size
    level = min(1.0, np.ceil((n + 1) * target) / n)
    return float(np.quantile(flagged, level, method="higher"))


def achieved_recall(scores: np.ndarray, unsupported: np.ndarray, threshold: float) -> float:
    flagged = np.asarray(scores)[np.asarray(unsupported, dtype=bool)]
    if flagged.size == 0:
        return float("nan")
    return float(np.mean(flagged <= threshold))


def false_alarm_rate(scores: np.ndarray, unsupported: np.ndarray, threshold: float) -> float:
    """The share of supported claims the threshold flags, which is what recall costs."""
    supported = np.asarray(scores)[~np.asarray(unsupported, dtype=bool)]
    if supported.size == 0:
        return float("nan")
    return float(np.mean(supported <= threshold))


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def covers(self, value: float) -> bool:
        return self.low <= value <= self.high


def bootstrap_recall(scores: np.ndarray, unsupported: np.ndarray, threshold: float,
                     draws: int = 2000, seed: int = 0) -> Interval:
    """A percentile interval on achieved recall, resampling the unsupported claims."""
    flagged = np.asarray(scores)[np.asarray(unsupported, dtype=bool)]
    if flagged.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    rng = RNG(seed)
    idx = rng.integers(0, flagged.size, size=(draws, flagged.size))
    draws_recall = np.mean(flagged[idx] <= threshold, axis=1)
    return Interval(
        float(np.mean(flagged <= threshold)),
        float(np.quantile(draws_recall, 0.025)),
        float(np.quantile(draws_recall, 0.975)),
    )


def density_ratio(source_features: np.ndarray, target_features: np.ndarray,
                  seed: int = 0) -> np.ndarray:
    """How much each source claim looks like a target claim.

    A logistic classifier separates source from target; the ratio ``p / (1 - p)`` of its
    probability is the importance weight under covariate shift. Weights are clipped, because
    one claim with a huge weight is a threshold chosen by one claim.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.vstack([np.asarray(source_features), np.asarray(target_features)])
    y = np.r_[np.zeros(len(source_features)), np.ones(len(target_features))]
    scaled = StandardScaler().fit_transform(x)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(scaled, y)
    p = model.predict_proba(scaled[: len(source_features)])[:, 1]
    weights = p / np.clip(1 - p, 1e-6, None)
    return np.asarray(np.clip(weights, 0.0, np.quantile(weights, 0.99)), dtype=float)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    order = np.argsort(values)
    values, weights = np.asarray(values)[order], np.asarray(weights)[order]
    total = weights.sum()
    if total <= 0:
        raise ValueError("weights must be positive somewhere")
    cumulative = np.cumsum(weights) / total
    return float(values[np.searchsorted(cumulative, level, side="left").clip(0, len(values) - 1)])


def weighted_threshold(scores: np.ndarray, unsupported: np.ndarray, weights: np.ndarray,
                       target: float) -> float:
    """``threshold_for_recall`` with each source claim weighted towards the target domain."""
    mask = np.asarray(unsupported, dtype=bool)
    flagged, w = np.asarray(scores)[mask], np.asarray(weights)[mask]
    if flagged.size == 0:
        raise ValueError("no unsupported claims to calibrate on")
    return weighted_quantile(flagged, w, target)


def mondrian_thresholds(scores: np.ndarray, unsupported: np.ndarray, groups: np.ndarray,
                        target: float, minimum: int = 25) -> dict[str, float]:
    """One threshold per group, falling back to the pooled one where a group is too small."""
    groups = np.asarray(groups)
    pooled = threshold_for_recall(scores, unsupported, target)
    out = {"__pooled__": pooled}
    for group in np.unique(groups):
        mask = groups == group
        if int(np.sum(np.asarray(unsupported, dtype=bool)[mask])) < minimum:
            continue
        out[str(group)] = threshold_for_recall(scores[mask], unsupported[mask], target)
    return out


def apply_mondrian(thresholds: dict[str, float], groups: np.ndarray) -> np.ndarray:
    pooled = thresholds["__pooled__"]
    return np.array([thresholds.get(str(g), pooled) for g in np.asarray(groups)])


def labels_to_recover(scores: np.ndarray, unsupported: np.ndarray, target: float,
                      budgets: tuple[int, ...] = (25, 50, 100, 200, 400),
                      tolerance: float = 0.02, draws: int = 50,
                      seed: int = 0) -> dict[int, dict[str, float]]:
    """Label ``k`` target claims, recalibrate on them, and measure on the rest.

    Returns, per budget, the mean achieved recall and the share of draws that land within
    ``tolerance`` of the promise. The claims used for recalibration are never scored.
    """
    scores, unsupported = np.asarray(scores), np.asarray(unsupported, dtype=bool)
    out: dict[int, dict[str, float]] = {}
    for k in budgets:
        if k >= len(scores):
            continue
        recalls, within, alarms = [], [], []
        for draw in range(draws):
            rng = RNG(seed + 1000 * draw)
            idx = rng.choice(len(scores), k, replace=False)
            rest = np.setdiff1d(np.arange(len(scores)), idx)
            if unsupported[idx].sum() < 5:
                continue  # too few positives in the sample to calibrate on
            threshold = threshold_for_recall(scores[idx], unsupported[idx], target)
            recall = achieved_recall(scores[rest], unsupported[rest], threshold)
            recalls.append(recall)
            within.append(float(recall >= target - tolerance))
            alarms.append(false_alarm_rate(scores[rest], unsupported[rest], threshold))
        if recalls:
            out[k] = {
                "recall": float(np.mean(recalls)),
                "recall_sd": float(np.std(recalls)),
                "within_tolerance": float(np.mean(within)),
                "false_alarm_rate": float(np.mean(alarms)),
                "draws": float(len(recalls)),
            }
    return out


def psi(source: np.ndarray, target: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two score distributions."""
    edges = np.quantile(np.asarray(source), np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    s = np.histogram(source, edges)[0] / max(len(source), 1)
    t = np.histogram(target, edges)[0] / max(len(target), 1)
    s, t = np.clip(s, 1e-6, None), np.clip(t, 1e-6, None)
    return float(np.sum((t - s) * np.log(t / s)))


def mmd(source: np.ndarray, target: np.ndarray, permutations: int = 200, seed: int = 0,
        max_n: int = 800) -> tuple[float, float]:
    """RBF-kernel MMD^2 between two score samples, with a permutation p-value."""
    rng = RNG(seed)
    a = np.asarray(source).reshape(-1, 1)
    b = np.asarray(target).reshape(-1, 1)
    a = a[rng.choice(len(a), min(len(a), max_n), replace=False)]
    b = b[rng.choice(len(b), min(len(b), max_n), replace=False)]
    z = np.vstack([a, b])
    d2 = (z - z.T) ** 2
    positive = d2[d2 > 0]
    gamma = 1.0 / np.median(positive) if positive.size else 1.0
    k = np.exp(-gamma * d2)
    n = len(a)

    def stat(idx: np.ndarray) -> float:
        x, y = idx[:n], idx[n:]
        return float(k[np.ix_(x, x)].mean() + k[np.ix_(y, y)].mean() - 2 * k[np.ix_(x, y)].mean())

    observed = stat(np.arange(len(z)))
    null = [stat(rng.permutation(len(z))) for _ in range(permutations)]
    return observed, float((1 + sum(v >= observed for v in null)) / (1 + permutations))
