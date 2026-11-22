"""Fill the README's results section from what ``shiftbench analyse`` wrote.

    python scripts/render_readme.py            # rewrite the section
    python scripts/render_readme.py --check    # fail if it is out of date

The numbers in a README drift away from the files that produced them, and the drift is
always flattering, so the section is generated and `make readme-check` fails when it does
not match ``results/``. Written after a sibling project's hand-kept README was found
claiming its release gate rejected a model the gate in fact promoted.

The marker is ``RESULTS_SECTION`` on a first run and the comment pair it is replaced by
afterwards, so re-rendering is idempotent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

README = Path("README.md")
RESULTS = Path("results")
MARKER = "RESULTS"
TOLERANCE = 0.02


def load() -> dict[str, Any]:
    missing = [
        name
        for name in ("leaderboard", "pairs", "budgets", "drift")
        if not (RESULTS / f"{name}.parquet").exists()
    ]
    if missing or not (RESULTS / "summary.json").exists():
        raise SystemExit(
            "results/ is incomplete: run `make score` then `make analyse` first "
            f"(missing {', '.join(missing) or 'summary.json'})"
        )
    return {
        "leaderboard": pd.read_parquet(RESULTS / "leaderboard.parquet"),
        "pairs": pd.read_parquet(RESULTS / "pairs.parquet"),
        "budgets": pd.read_parquet(RESULTS / "budgets.parquet"),
        "drift": pd.read_parquet(RESULTS / "drift.parquet"),
        "summary": json.loads((RESULTS / "summary.json").read_text()),
    }


def leaderboard(data: dict[str, Any]) -> list[str]:
    board, summary = data["leaderboard"], data["summary"]
    promised = float(summary["promised"])
    lines = [
        f"Every threshold promises {promised:.0%} recall on unsupported claims, fitted on one "
        f"domain and used on another. {len(summary['domains'])} domains give "
        f"{len(summary['domains']) * (len(summary['domains']) - 1)} ordered pairs, over "
        f"{summary['claims']:,} claims in all.",
        "",
        "| detector | promise kept | mean gap | worst gap | false alarms | labels to recover |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in board.itertuples(index=False):
        labels = "n/a" if r.labels_to_recover != r.labels_to_recover else f"{int(r.labels_to_recover)}"
        lines.append(
            f"| {r.detector} | {r.promise_kept:.0%} | {r.mean_gap:+.3f} | {r.worst_gap:+.3f} | "
            f"{r.mean_false_alarms:.2f} | {labels} |"
        )
    lines += [
        "",
        '"Promise kept" is the share of ordered pairs whose achieved recall came within '
        f"{TOLERANCE:.2f} of what was promised. The gap is achieved minus promised, so it is "
        "good near zero and a large positive one means the detector flags more than it was "
        "asked to. That is why detectors keeping as many promises are ranked by what the "
        "promise cost in false alarms rather than by the gap.",
        "",
    ]
    return lines


def matrix(data: dict[str, Any]) -> list[str]:
    pairs = data["pairs"]
    promised = float(data["summary"]["promised"])
    split = pairs[pairs["method"] == "split"]
    lines = []
    for detector, group in split.groupby("detector"):
        table = group.pivot_table(index="source", columns="target", values="achieved")
        domains = list(table.columns)
        lines += [
            f"**{detector}**, recall achieved when the threshold is fitted on the row's domain "
            f"and used on the column's, against a promise of {promised:.0%}. The diagonal is a "
            "held-out half of the same domain, so it is a measurement and not a tautology.",
            "",
            "| fitted on | " + " | ".join(domains) + " |",
            "| --- | " + " | ".join("---" for _ in domains) + " |",
        ]
        for source in table.index:
            cells = []
            for target in domains:
                value = table.loc[source, target]
                cells.append("n/a" if value != value else f"{value:.2f}")
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def methods(data: dict[str, Any]) -> list[str]:
    pairs = data["pairs"]
    off = pairs[pairs["source"] != pairs["target"]]
    promised = float(data["summary"]["promised"])
    lines = [
        "### What the repairs are worth",
        "",
        "The same pairs under each way of choosing the threshold. `oracle` is fitted on the "
        "target's own labels and is the ceiling nobody has; `split` is what a team ships.",
        "",
        "| method | mean recall | within tolerance | mean false alarms |",
        "| --- | --- | --- | --- |",
    ]
    for method in ("split", "weighted", "mondrian", "oracle"):
        rows = off[off["method"] == method]
        if rows.empty:
            continue
        kept = float((rows["achieved"] >= promised - TOLERANCE).mean())
        lines.append(
            f"| {method} | {rows['achieved'].mean():.3f} | {kept:.0%} | {rows['false_alarms'].mean():.2f} |"
        )
    weighted = off[off["method"] == "weighted"]
    if not weighted.empty and weighted["weight_spread"].notna().any():
        spread = float(weighted["weight_spread"].mean())
        separable = float((weighted["domain_separability"] >= 0.999).mean())
        split_recall = float(off[off["method"] == "split"]["achieved"].mean())
        moved = float(weighted["achieved"].mean()) - split_recall
        lines += [
            "",
            f"The weighted row comes with diagnostics, because there are two different "
            f"reasons it could sit near the split row. The domain classifier separated the "
            f"two domains perfectly on {separable:.0%} of pairs, and the mean spread of its "
            f"importance weights was {spread:.2f}.",
        ]
        # Degeneracy and a genuine null result look identical in the recall column and are
        # not the same finding, so the diagnostics decide which one is being reported.
        if spread < 0.05 or separable > 0.5:
            lines.append(
                "That is the degenerate case: importance weighting needs the two domains to "
                "overlap, and where they do not every source claim gets near enough the same "
                "weight, so the weighted threshold is the plain one under another name. The "
                "repair did not fail here, it never ran."
            )
        else:
            lines.append(
                f"So the weights did vary, and the reweighting still moved mean recall by "
                f"{moved:+.3f}. That is a real negative result rather than a degenerate "
                "estimator: the density ratio had something to work with and reweighting the "
                "source towards the target did not recover the recall that the shift cost."
            )
    lines.append("")
    return lines


def repair(data: dict[str, Any]) -> list[str]:
    curves, promised = data["budgets"], float(data["summary"]["promised"])
    lines = [
        "### What it costs to get the promise back",
        "",
        f"Label k claims from the new domain, refit the threshold on them, measure on the "
        f"rest, over many draws. Averaged over domains and detectors, against a promise of "
        f"{promised:.0%}.",
        "",
        "| labels | mean recall | spread | share of draws within tolerance | false alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_budget = curves.groupby("labels")
    for labels, group in by_budget:
        lines.append(
            f"| {int(labels)} | {group['recall'].mean():.3f} | {group['recall_sd'].mean():.3f} | "
            f"{group['within_tolerance'].mean():.0%} | {group['false_alarm_rate'].mean():.2f} |"
        )
    means = by_budget["recall"].mean()
    spreads = by_budget["recall_sd"].mean()
    first, last = int(means.index[0]), int(means.index[-1])
    lines += [
        "",
        f"The column that moves is the spread, from {spreads.iloc[0]:.3f} at {first} labels to "
        f"{spreads.iloc[-1]:.3f} at {last}. Mean recall does not climb with the budget and is "
        "not supposed to: refitting on the target's own claims puts the threshold in the right "
        "place at every budget, and what more labels buy is a threshold that lands there "
        "reliably rather than one that happens to. A small sample gives a noisy threshold, and "
        "a noisy threshold that errs towards flagging looks like good recall on average while "
        "being wrong in both directions on any given draw.",
        "",
    ]
    return lines


def drift(data: dict[str, Any]) -> list[str]:
    summary = data["summary"]["drift_predicts_loss"]
    if "psi_spearman" not in summary:
        return []
    lines = [
        "### Did anything warn before the labels arrived",
        "",
        "PSI and MMD on the score distributions need no target labels, so they are the only "
        "warning available in advance. Spearman correlation against the recall the shipped "
        f"threshold lost, over {int(summary['pairs'])} ordered pairs.",
        "",
        "| statistic | against signed loss | against absolute loss | among pairs that broke |",
        "| --- | --- | --- | --- |",
    ]
    for name in ("psi", "mmd"):
        lines.append(
            f"| {name.upper()} | {summary[f'{name}_spearman']:+.2f} | "
            f"{summary[f'{name}_spearman_absolute']:+.2f} | "
            f"{summary[f'{name}_spearman_undercovering']:+.2f} |"
        )
    lines += [
        "",
        f"The last column is the one to read, over the {int(summary['undercovering_pairs'])} "
        "pairs where the promise actually broke. The first is near zero or negative, and that "
        "is arithmetic rather than a result: a score-distribution statistic is close to "
        "direction-blind, coming out nearly the same for a pair and for its reverse, while "
        "the loss does not. Shipping a threshold from an easy domain to a hard one loses "
        "recall and the reverse over-recalls, which pins the loss at its floor however large "
        "the drift, so correlating over both directions mixes a relationship with its mirror "
        "image. A high reading means recalibrate and measure, not that the promise moved in "
        "a particular direction.",
        "",
    ]
    return lines


def body(data: dict[str, Any]) -> str:
    lines = ["## Results", ""]
    for part in (leaderboard, matrix, methods, repair, drift):
        lines += part(data)
    return "\n".join(lines).rstrip() + "\n"


def replace(text: str, rendered: str) -> str:
    start, end = f"<!-- {MARKER}:start -->", f"<!-- {MARKER}:end -->"
    block = f"{start}\n{rendered}{end}"
    if start in text and end in text:
        return text[: text.index(start)] + block + text[text.index(end) + len(end) :]
    return text.replace(f"{MARKER}_SECTION", block)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the README is out of date")
    args = ap.parse_args()

    original = README.read_text()
    text = replace(original, body(load()))
    if f"{MARKER}_SECTION" in text:
        raise SystemExit(f"README has no {MARKER} marker to fill")
    if args.check:
        if text != original:
            raise SystemExit("README does not match results/: run `make readme`")
        print("README matches results/")
        return
    README.write_text(text)
    print(f"rendered the results section into {README}")


if __name__ == "__main__":
    main()
