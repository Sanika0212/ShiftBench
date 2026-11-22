"""The one claim format every loader has to answer to."""

from __future__ import annotations

from pathlib import Path

import pytest

from shiftbench.domains import COLUMNS, DOMAINS, frame, load, unified


def rows(n: int = 3, domain: str = "legal", start: int = 0) -> list[dict[str, object]]:
    return [
        {
            "claim_id": f"{domain}-{i}",
            "domain": domain,
            "claim": f"claim {i}",
            "evidence": f"evidence {i}",
            "supported": i % 2 == 0,
            "group_id": f"doc-{i // 2}",
            "source": domain,
        }
        for i in range(start, start + n)
    ]


def test_every_domain_declares_a_licence_and_whether_it_may_be_shared() -> None:
    """Two of the five are non-commercial, so this has to be explicit per domain."""
    assert len(DOMAINS) == 5
    for key, domain in DOMAINS.items():
        assert domain.key == key
        assert domain.licence
        assert isinstance(domain.redistributable, bool)
        assert domain.files
    non_commercial = [k for k, d in DOMAINS.items() if not d.redistributable]
    assert set(non_commercial) == {"scientific", "medical", "legal"}


def test_a_frame_keeps_the_columns_and_coerces_the_label() -> None:
    df = frame(rows(4))
    assert list(df.columns) == COLUMNS
    assert df["supported"].dtype == bool
    assert len(df) == 4


def test_duplicate_claim_ids_are_refused() -> None:
    """A repeated id would silently overwrite a score in the cache, which is keyed by id."""
    with pytest.raises(ValueError, match="unique"):
        frame(rows(2) + rows(2))


def test_an_unknown_domain_says_which_ones_exist() -> None:
    with pytest.raises(KeyError, match="unknown domain"):
        load("chemical", Path("/nonexistent"))


def test_a_domain_that_was_never_fetched_says_how_to_fetch_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="shiftbench fetch --domain legal"):
        load("legal", tmp_path)


def test_unified_prefers_the_combined_file_when_it_exists(tmp_path: Path) -> None:
    combined = frame(rows(2, "legal") + rows(2, "financial"))
    combined.to_parquet(tmp_path / "claims.parquet", index=False)
    out = unified(tmp_path)
    assert len(out) == 4
    assert set(out["domain"]) == {"legal", "financial"}
