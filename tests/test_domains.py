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


def test_a_claim_with_no_evidence_is_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    """Fin-Fact ships 101 of these in 2,758, and a detector's score on one is an artifact.

    Word overlap returns 0.0 for an empty passage, which reads as maximally unsupported
    whatever the label says, so these rows are noise with a label attached.
    """
    rows = [
        {
            "claim_id": f"d-{i}",
            "domain": "financial",
            "claim": claim,
            "evidence": evidence,
            "supported": True,
            "group_id": "g",
            "source": "s",
        }
        for i, (claim, evidence) in enumerate(
            [("a real claim", "real evidence"), ("no evidence here", "   "), ("", "orphaned")]
        )
    ]
    out = frame(rows)
    assert list(out["claim_id"]) == ["d-0"]
    assert "dropped 2 of 3" in capsys.readouterr().out


def test_ids_stay_stable_when_a_row_is_dropped() -> None:
    """The ids are assigned before the drop, so a surviving row keeps the id it had."""
    rows = [
        {
            "claim_id": f"d-{i}",
            "domain": "financial",
            "claim": "" if i == 1 else "claim",
            "evidence": "evidence",
            "supported": i % 2 == 0,
            "group_id": "g",
            "source": "s",
        }
        for i in range(4)
    ]
    assert list(frame(rows)["claim_id"]) == ["d-0", "d-2", "d-3"]
