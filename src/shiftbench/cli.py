"""Command line for the benchmark.

    shiftbench fetch --limit 600            # every domain, capped, into data/
    shiftbench score --detector minicheck   # chunked, resumable, duty cycled
    shiftbench analyse                      # matrices, budgets, drift, leaderboard
    shiftbench serve --port 8090            # the leaderboard site over the results

Scoring is the only expensive step. It can be stopped at any point and started again: the
per-claim cache means nothing already scored is scored twice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shiftbench.detectors import DEFAULT_DETECTORS
from shiftbench.domains import DOMAINS

RESULTS = Path("results")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="shiftbench", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fetch", help="download the domains and write the unified claims")
    f.add_argument("--domain", action="append", choices=sorted(DOMAINS), default=None)
    f.add_argument("--limit", type=int, default=600, help="claims per domain, 0 for all")
    f.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("score", help="score claims with a detector, resuming the cache")
    s.add_argument("--domain", action="append", choices=sorted(DOMAINS), default=None)
    s.add_argument("--detector", action="append", default=None)
    s.add_argument("--chunk", type=int, default=32)
    s.add_argument("--duty", type=float, default=0.4, help="share of each second spent working")
    s.add_argument("--models", type=Path, default=None)

    a = sub.add_parser("analyse", help="the source by target matrices and the leaderboard")
    a.add_argument("--detector", action="append", default=None)
    a.add_argument("--promised", type=float, default=0.90)
    a.add_argument("--seed", type=int, default=0)

    v = sub.add_parser("serve", help="the leaderboard site")
    v.add_argument("--port", type=int, default=8090)
    v.add_argument("--host", default="127.0.0.1")

    args = ap.parse_args(argv)

    if args.command == "fetch":
        from shiftbench.fetch import build

        for domain in args.domain or sorted(DOMAINS):
            claims = build(domain, limit=args.limit, seed=args.seed)
            share = float(claims["supported"].mean())
            print(f"{domain}: {len(claims)} claims, {share:.0%} supported, "
                  f"licence {DOMAINS[domain].licence}")
    elif args.command == "score":
        from shiftbench.domains import load
        from shiftbench.runner import score_domain

        for detector in args.detector or list(DEFAULT_DETECTORS):
            for domain in args.domain or sorted(DOMAINS):
                claims = load(domain)
                score_domain(claims, detector, chunk=args.chunk, duty=args.duty,
                             models=args.models)
    elif args.command == "analyse":
        import pandas as pd

        from shiftbench.analysis import budgets, drift, drift_predicts_loss, leaderboard, matrix
        from shiftbench.domains import unified
        from shiftbench.runner import scored_frame

        claims = unified()
        scored = scored_frame(claims, args.detector or list(DEFAULT_DETECTORS))
        scored["unsupported"] = ~scored["supported"]
        pairs = matrix(scored, promised=args.promised, seed=args.seed)
        curves = budgets(scored, promised=args.promised, seed=args.seed)
        drifts = drift(scored, seed=args.seed)
        board = leaderboard(pairs, curves)
        RESULTS.mkdir(parents=True, exist_ok=True)
        for name, table in (("pairs", pairs), ("budgets", curves), ("drift", drifts),
                            ("leaderboard", board)):
            table.to_parquet(RESULTS / f"{name}.parquet", index=False)
        summary = {
            "promised": args.promised,
            "detectors": sorted(scored["detector"].unique()),
            "domains": sorted(scored["domain"].unique()),
            "claims": int(len(claims)),
            "drift_predicts_loss": drift_predicts_loss(pairs, drifts),
        }
        (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
        with pd.option_context("display.width", 140):
            print(board.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(json.dumps(summary["drift_predicts_loss"], indent=2))
        print("wrote", RESULTS)
    elif args.command == "serve":
        import uvicorn

        from shiftbench.site import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
