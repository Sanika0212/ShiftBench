"""Score every claim once, in chunks that can be stopped and resumed.

Scoring is the only expensive part of this benchmark, and it runs on a laptop, so it is
built to be interrupted. Each chunk of claims is scored, appended to a JSONL cache and
followed by an idle pause; starting the runner again skips everything already in the cache.
Stopping it costs at most one chunk.

    shiftbench score --domain legal --detector minicheck --limit 600

Scores from another project can be imported instead of recomputed: Evidence Gate keeps
per-item scores for the three domains it evaluated, and ``import_scores`` brings them in
under the same cache format.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shiftbench.detectors import Detector, build
from shiftbench.thermal import (
    MIN_AVAILABLE,
    DutyCycle,
    background_qos,
    limit_threads,
    wait_for_memory,
)

SCORES = Path("data/scores")


def cache_path(domain: str, detector: str, root: Path = SCORES) -> Path:
    return root / f"{domain}-{detector}.jsonl"


def read_cache(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    out: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["claim_id"]] = float(row["support"])
    return out


def chunks(frame: pd.DataFrame, size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


def score_domain(
    claims: pd.DataFrame,
    detector_name: str,
    detector: Detector | None = None,
    chunk: int = 32,
    duty: float = 0.4,
    root: Path = SCORES,
    models: Path | None = None,
    progress: bool = True,
    min_available: float = MIN_AVAILABLE,
    background: bool = False,
) -> pd.DataFrame:
    """Score one domain with one detector, resuming from whatever is already cached.

    ``min_available`` is the share of memory the kernel must still consider available
    before each chunk; below it the loop waits rather than pushing the machine into swap.
    ``background`` asks for background scheduling, which is off by default because it costs
    far more wall clock than it saves in heat.
    """
    limit_threads()
    if background:
        background_qos()
    domain = str(claims["domain"].iloc[0])
    path = cache_path(domain, detector_name, root)
    done = read_cache(path)
    todo = claims[~claims["claim_id"].isin(done)]
    if len(todo):
        detector = detector or build(detector_name, models)
        path.parent.mkdir(parents=True, exist_ok=True)
        cycle = DutyCycle(duty)
        waited = 0.0
        for i, batch in enumerate(chunks(todo, chunk), start=1):
            if min_available:
                # Another application may have taken the memory since the last chunk.
                waited += wait_for_memory(min_available)
            with cycle:
                scores = detector.score(batch["claim"].tolist(), batch["evidence"].tolist())
            with path.open("a") as fh:
                for claim_id, support in zip(batch["claim_id"], scores, strict=True):
                    fh.write(json.dumps({"claim_id": claim_id, "support": float(support)}) + "\n")
            if progress and (i % 5 == 0 or i * chunk >= len(todo)):
                scored = min(i * chunk, len(todo))
                print(
                    f"  {domain}/{detector_name}: {scored}/{len(todo)} "
                    f"(worked {cycle.worked / 60:.1f} min, idle {cycle.paused / 60:.1f} min"
                    + (f", waited {waited / 60:.1f} min for memory" if waited else "")
                    + ")",
                    flush=True,
                )
        done = read_cache(path)
    out = claims.copy()
    out["support"] = [done[c] for c in out["claim_id"]]
    out["detector"] = detector_name
    return out


def import_scores(rows: list[dict[str, Any]], domain: str, detector_name: str, root: Path = SCORES) -> int:
    """Bring per-item scores computed elsewhere into the cache, keyed the same way."""
    path = cache_path(domain, detector_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Seen, not just already cached: the rows being imported can repeat an id themselves.
    # SciFact cites one document twice for claim 1245, so its 1259 cached scores cover 1258
    # pairs, and checking only the existing cache wrote that one twice and counted it twice.
    seen = set(read_cache(path))
    added = 0
    with path.open("a") as fh:
        for row in rows:
            claim_id = str(row["claim_id"])
            if claim_id in seen:
                continue
            seen.add(claim_id)
            fh.write(json.dumps({"claim_id": claim_id, "support": float(row["support"])}) + "\n")
            added += 1
    return added


def scored_frame(claims: pd.DataFrame, detectors: list[str], root: Path = SCORES) -> pd.DataFrame:
    """Long frame of claim by detector, for every detector that has a complete cache."""
    frames = []
    for detector in detectors:
        parts = []
        for domain, group in claims.groupby("domain"):
            done = read_cache(cache_path(str(domain), detector, root))
            missing = int((~group["claim_id"].isin(done)).sum())
            if missing:
                print(f"  skipping {domain}/{detector}: {missing} claims not scored yet")
                break
            part = group.copy()
            part["support"] = [done[c] for c in part["claim_id"]]
            part["detector"] = detector
            parts.append(part)
        else:
            frames.append(pd.concat(parts, ignore_index=True))
    if not frames:
        raise RuntimeError("no detector has a complete set of scores yet")
    return pd.concat(frames, ignore_index=True)


def features(frame: pd.DataFrame) -> np.ndarray:
    """What the domain classifier sees: the score and two cheap properties of the pair."""
    return np.column_stack(
        [
            frame["support"].to_numpy(dtype=float),
            frame["claim"].str.len().to_numpy(dtype=float),
            frame["evidence"].str.len().to_numpy(dtype=float),
        ]
    )
