"""Bring in per-item detector scores that Evidence Gate already computed.

Evidence Gate scored SciFact and MedHallu with the same ONNX checkers this benchmark uses,
on the same claims, and kept the per-item scores. Reusing them is not a shortcut around the
measurement: it is the same model on the same pairs, and it saves hours of laptop heat.

    python scripts/import_evidence_gate.py --from ~/Projects/Evidence-Gate

Both repositories key those two domains the same way, so a score is matched by its id and
then checked against the label this benchmark holds for that claim. A score whose id is
unknown here, or whose label disagrees, is reported and dropped rather than guessed at.
RAGTruth is not imported: Evidence Gate scored it one answer sentence at a time, and this
benchmark treats a whole answer as the claim, so those numbers are not the same measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shiftbench.domains import load
from shiftbench.runner import import_scores

# Evidence Gate's file prefix -> (domain here, its detector name -> ours)
SOURCES = {
    "scifact": (
        "scientific",
        {
            "minicheck-int8": "minicheck",
            "nli-int8": "nli",
            "embedding-similarity": "embedding-similarity",
            "word-overlap": "word-overlap",
        },
    ),
    "medhallu": (
        "medical",
        {
            "minicheck-int8": "minicheck",
            "embedding-similarity": "embedding-similarity",
            "word-overlap": "word-overlap",
        },
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from",
        dest="source",
        type=Path,
        required=True,
        help="an Evidence Gate checkout holding runs/scores",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    scores_dir = args.source.expanduser() / "runs" / "scores"
    if not scores_dir.exists():
        raise SystemExit(f"{scores_dir} does not exist")

    imported = 0
    for prefix, (domain, detectors) in SOURCES.items():
        try:
            claims = load(domain)
        except FileNotFoundError:
            print(f"{domain}: not fetched yet, skipping")
            continue
        labels = dict(zip(claims["claim_id"], ~claims["supported"], strict=True))
        for their_name, our_name in detectors.items():
            path = scores_dir / f"{prefix}-{their_name}.json"
            if not path.exists():
                # Named rather than skipped in silence: a renamed export over there would
                # otherwise look exactly like a detector this benchmark never scored.
                print(f"{domain}/{our_name}: {path.name} not found, nothing imported")
                continue
            rows: list[dict[str, Any]] = json.loads(path.read_text())
            matched, unknown, disagreed = [], 0, 0
            for row in rows:
                claim_id = str(row["id"])
                if claim_id not in labels:
                    unknown += 1
                    continue
                if bool(row["unsupported"]) != bool(labels[claim_id]):
                    disagreed += 1
                    continue
                matched.append({"claim_id": claim_id, "support": row["support"]})
            added = 0 if args.dry_run else import_scores(matched, domain, our_name)
            imported += added
            # Most of "unknown" is normally this benchmark's per-domain cap rather than a
            # key that does not line up, so the size of the sample is printed beside it.
            # A count near zero against a full sample is the signal worth noticing.
            print(
                f"{domain}/{our_name}: {len(matched)} of {len(rows)} scores matched into "
                f"{len(labels)} sampled claims ({unknown} not in the sample or unknown, "
                f"{disagreed} label mismatch), {added} new"
            )
            if matched and len(matched) < min(len(rows), len(labels)) // 2:
                print(
                    f"  warning: fewer than half of the {min(len(rows), len(labels))} "
                    "overlappable claims matched, which looks like a key mismatch"
                )
    print(f"imported {imported} scores" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
