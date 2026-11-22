"""Five domains of grounded claims, in one format.

A row is a claim, the evidence a reader is meant to check it against, and whether the claim
is supported by that evidence. The detectors under test score the pair; everything else in
this benchmark is about what happens to a threshold when the domain changes underneath it.

| key | source | what a claim is | licence |
| --- | --- | --- | --- |
| scientific | SciFact (bigbio mirror) | a claim against the abstract a reviewer cited | CC BY-NC 2.0 |
| medical | MedHallu (PubMedQA) | an answer sentence against the PubMed context | see card |
| general | RAGTruth | an answer sentence against the retrieved passages | MIT |
| legal | ContractNLI | a hypothesis about a contract against its spans | CC BY-NC-SA 4.0 |
| financial | Fin-Fact | a financial claim against the evidence a fact checker used | Apache 2.0 |

Three of the five are not redistributed here: SciFact and ContractNLI carry non-commercial
licences, and MedHallu's terms have to be read off its dataset card. So this repository ships
loaders and cached detector scores, never the raw text. ``shiftbench fetch`` downloads the
sources into ``data/`` from pinned revisions; nothing here redistributes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("data")
UNIFIED = ROOT / "claims.parquet"

# A claim row: claim, evidence, supported (bool), domain, source id, and a group id that
# keeps claims from one document together when a split has to be drawn.
COLUMNS = ["claim_id", "domain", "claim", "evidence", "supported", "group_id", "source"]

# Most of these datasets ship a loading script rather than data files, and datasets 3.x will
# not run scripts. The Hub's automatic Parquet conversion gives the same rows as plain files,
# on a branch of its own, so that is what gets downloaded. Checked by HTTP status: every path
# below answers 200 on the revision beside it.
PARQUET = "refs/convert/parquet"


@dataclass(frozen=True)
class Domain:
    key: str
    repo: str
    revision: str
    files: tuple[str, ...]
    licence: str
    redistributable: bool


DOMAINS: dict[str, Domain] = {
    "scientific": Domain(
        key="scientific",
        # bigbio's mirror, not allenai/scifact: the original ships a loading script and no
        # data files, and datasets 3.x no longer runs scripts. This is also the mirror
        # Evidence Gate scored, so the claim ids here match the scores imported from it.
        repo="bigbio/scifact",
        revision=PARQUET,
        files=(
            "scifact_claims_source/train/0000.parquet",
            "scifact_claims_source/validation/0000.parquet",
            "scifact_corpus_source/train/0000.parquet",
        ),
        licence="CC BY-NC 2.0",
        redistributable=False,
    ),
    "medical": Domain(
        key="medical",
        repo="UTAustin-AIHealth/MedHallu",
        revision=PARQUET,
        files=("pqa_labeled/train/0000.parquet",),
        licence="see dataset card",
        redistributable=False,
    ),
    "general": Domain(
        key="general",
        repo="wandb/RAGTruth-processed",
        revision=PARQUET,
        files=("default/train/0000.parquet", "default/test/0000.parquet"),
        licence="MIT",
        redistributable=True,
    ),
    "legal": Domain(
        key="legal",
        repo="kiddothe2b/contract-nli",
        revision=PARQUET,
        files=("contractnli_a/train/0000.parquet", "contractnli_a/test/0000.parquet"),
        licence="CC BY-NC-SA 4.0",
        redistributable=False,
    ),
    "financial": Domain(
        key="financial",
        repo="amanrangapur/Fin-Fact",
        revision="main",  # a real file in the repository, not an auto conversion
        files=("finfact.json",),
        licence="Apache 2.0",
        redistributable=True,
    ),
}


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Every loader returns this shape, so the runner never knows which domain it has."""
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["supported"] = df["supported"].astype(bool)
    # A claim with nothing to check it against is not a grounding task, and a detector's
    # score on one is an artifact of how it handles the empty string rather than a judgement:
    # word overlap returns 0.0 for every one of them, which reads as maximally unsupported
    # whatever the label says. Fin-Fact ships 24 of these in 600. Dropped here, after the
    # ids are assigned, so an id stays stable whether or not its row survives.
    usable = df["claim"].str.strip().str.len().gt(0) & df["evidence"].str.strip().str.len().gt(0)
    if not usable.all():
        dropped = int((~usable).sum())
        name = str(df["domain"].iloc[0]) if len(df) else "?"
        print(f"  {name}: dropped {dropped} of {len(df)} claims with no claim or no evidence")
        df = df[usable].reset_index(drop=True)
    if df["claim_id"].duplicated().any():
        raise ValueError("claim ids must be unique within a domain")
    return df


def load(domain: str, root: Path = ROOT) -> pd.DataFrame:
    """Read one domain from whatever ``fetch`` left in ``root``."""
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain}; known: {sorted(DOMAINS)}")
    path = root / f"{domain}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `shiftbench fetch --domain {domain}`")
    return pd.read_parquet(path)


def unified(root: Path = ROOT) -> pd.DataFrame:
    """Every domain in one frame, which is what the analysis works on."""
    path = root / UNIFIED.name
    if path.exists():
        return pd.read_parquet(path)
    return pd.concat([load(d, root) for d in DOMAINS], ignore_index=True)
