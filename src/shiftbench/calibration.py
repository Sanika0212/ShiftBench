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


def corrected_level(n: int, target: float) -> float:
    """The quantile level that makes a promised recall of ``target`` hold on ``n`` claims.

    The conformal correction, ``ceil((n + 1) * target) / n``: on a new exchangeable sample
    the promise then holds with the stated probability rather than only in expectation. It
    matters most where the labels are fewest, which is exactly where this benchmark looks.

    Every scheme has to use this. It was applied to the split, mondrian and oracle
    thresholds and not to the weighted one, which handicapped the weighted scheme by an
    amount that has nothing to do with reweighting: measured with uniform weights, so the
    density ratio did nothing at all, 4.0 points of recall at 50 unsupported claims, 1.9 at
    100 and 0.7 at 300, against a tolerance of 2 points. A benchmark whose output is a
    ranking of these schemes cannot hold one of them to a different level.
    """
    return float(min(1.0, np.ceil((n + 1) * target) / n))


def threshold_for_recall(scores: np.ndarray, unsupported: np.ndarray, target: float) -> float:
    """The smallest threshold whose recall on these claims is at least ``target``.

    Scores are "how supported this claim looks", so a claim is flagged when its score falls
    below the threshold. A larger threshold flags more and recalls more, so the threshold
    that meets the promise at the least cost in false alarms is the smallest one that meets
    it, taken at the level ``corrected_level`` gives.
    """
    flagged = np.asarray(scores)[np.asarray(unsupported, dtype=bool)]
    if flagged.size == 0:
        raise ValueError("no unsupported claims to calibrate on")
    # The ceil((n + 1) * target)-th smallest, which is what the corrected level means. Taken
    # with numpy's "higher" this landed one position further up, because numpy spaces n
    # values over n - 1 intervals: at 50 claims that is 0.9215 achieved against a promised
    # 0.90, and it did not match the weighted scheme, which spaces them over n.
    n = flagged.size
    rank = int(np.ceil(corrected_level(n, target) * n))
    return float(np.sort(flagged)[min(n, max(1, rank)) - 1])


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


@dataclass(frozen=True)
class Reweighting:
    """Importance weights for the source claims, with what they are worth.

    ``separability`` is the domain classifier's AUC over both samples. At 1.0 the two
    domains do not overlap in these features, and an importance weight is then an
    extrapolation rather than an estimate: nothing in the source resembles the target, so
    there is no reweighting of the source that turns it into the target.

    ``spread`` is the standard deviation of the weights divided by their mean. Near zero the
    weights are all but equal and the weighted threshold is the unweighted one, whatever the
    classifier reports. This is the diagnostic that matters and it is not the effective
    sample size, which moves the wrong way: measured on separable domains, ESS rose to 94%
    of the sample while the threshold stopped moving, because near-equal weights look
    healthy by that measure and do nothing by this one.
    """

    weights: np.ndarray
    separability: float
    spread: float

    @property
    def degenerate(self) -> bool:
        """Whether the reweighting can be expected to do anything at all."""
        return self.spread < 0.05 or self.separability >= 0.999


def reweight(source_features: np.ndarray, target_features: np.ndarray,
             seed: int = 0) -> Reweighting:
    """How much each source claim looks like a target claim, and whether that means anything.

    A logistic classifier separates source from target; the ratio ``p / (1 - p)`` of its
    probability is the importance weight under covariate shift. Weights are clipped, because
    one claim with a huge weight is a threshold chosen by one claim.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    source_features = np.asarray(source_features)
    target_features = np.asarray(target_features)
    x = np.vstack([source_features, target_features])
    y = np.r_[np.zeros(len(source_features)), np.ones(len(target_features))]
    scaled = StandardScaler().fit_transform(x)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(scaled, y)
    both = model.predict_proba(scaled)[:, 1]
    p = both[: len(source_features)]
    weights = p / np.clip(1 - p, 1e-6, None)
    weights = np.asarray(np.clip(weights, 0.0, np.quantile(weights, 0.99)), dtype=float)
    mean = float(weights.mean())
    return Reweighting(
        weights=weights,
        separability=float(roc_auc_score(y, both)) if len(np.unique(y)) > 1 else float("nan"),
        spread=float(weights.std() / mean) if mean > 0 else 0.0,
    )


def density_ratio(source_features: np.ndarray, target_features: np.ndarray,
                  seed: int = 0) -> np.ndarray:
    """The weights alone, for callers that do not need the diagnostics."""
    return reweight(source_features, target_features, seed).weights


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
    """``threshold_for_recall`` with each source claim weighted towards the target domain.

    Takes the same corrected level as the unweighted schemes, so that a difference between
    them is a difference between the schemes. The textbook weighted conformal correction
    puts the mass of one more point at infinity and depends on the weights themselves; this
    uses the unweighted correction instead, which is the one the other schemes get.
    """
    mask = np.asarray(unsupported, dtype=bool)
    flagged, w = np.asarray(scores)[mask], np.asarray(weights)[mask]
    if flagged.size == 0:
        raise ValueError("no unsupported claims to calibrate on")
    return weighted_quantile(flagged, w, corrected_level(flagged.size, target))


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
