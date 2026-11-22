"""The command line: that each subcommand reaches the right code with the right arguments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from shiftbench import cli


def claims(domain: str = "scientific", n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "claim_id": [f"{domain}-{i}" for i in range(n)],
            "domain": domain,
            "claim": [f"claim {i}" for i in range(n)],
            "evidence": [f"evidence {i}" for i in range(n)],
            "supported": [i % 2 == 0 for i in range(n)],
            "group_id": [f"g{i % 2}" for i in range(n)],
            "source": domain,
        }
    )


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        cli.main([])


def test_fetch_builds_each_domain_and_reports_its_licence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[tuple[str, int, int]] = []

    def build(domain: str, limit: int = 0, seed: int = 0) -> pd.DataFrame:
        seen.append((domain, limit, seed))
        return claims(domain)

    monkeypatch.setattr("shiftbench.fetch.build", build)
    cli.main(["fetch", "--domain", "scientific", "--limit", "25", "--seed", "3"])
    assert seen == [("scientific", 25, 3)]
    out = capsys.readouterr().out
    assert "scientific: 8 claims" in out
    assert "50% supported" in out
    assert "licence" in out


def test_fetch_covers_every_domain_when_none_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    monkeypatch.setattr(
        "shiftbench.fetch.build",
        lambda domain, limit=0, seed=0: seen.append(domain) or claims(domain),  # type: ignore[func-returns-value]
    )
    cli.main(["fetch"])
    assert seen == sorted(cli.DOMAINS)


def test_score_runs_every_default_detector_over_every_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, int, float]] = []

    def score_domain(frame: pd.DataFrame, detector: str, **kw: Any) -> pd.DataFrame:
        calls.append((str(frame["domain"].iloc[0]), detector, kw["chunk"], kw["duty"]))
        return frame

    monkeypatch.setattr("shiftbench.domains.load", lambda domain: claims(domain))
    monkeypatch.setattr("shiftbench.runner.score_domain", score_domain)
    cli.main(["score", "--domain", "medical", "--detector", "word-overlap", "--chunk", "4"])
    assert calls == [("medical", "word-overlap", 4, 0.4)]


def test_analyse_writes_every_table_and_a_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The four parquet tables and summary.json are what the site and the warehouse read."""
    scored = pd.concat([claims("scientific", 40), claims("medical", 40)], ignore_index=True)
    scored["support"] = [0.2 if not s else 0.8 for s in scored["supported"]]
    scored["detector"] = "word-overlap"

    monkeypatch.setattr("shiftbench.domains.unified", lambda: scored)
    monkeypatch.setattr("shiftbench.runner.scored_frame", lambda c, d: scored)
    monkeypatch.setattr(cli, "RESULTS", tmp_path / "results")

    cli.main(["analyse", "--detector", "word-overlap"])

    for name in ("pairs", "budgets", "drift", "leaderboard"):
        assert (tmp_path / "results" / f"{name}.parquet").exists(), name
    summary = json.loads((tmp_path / "results" / "summary.json").read_text())
    assert summary["promised"] == 0.90
    assert summary["detectors"] == ["word-overlap"]
    assert sorted(summary["domains"]) == ["medical", "scientific"]
    assert "drift_predicts_loss" in summary
    assert "wrote" in capsys.readouterr().out


def test_serve_hands_the_app_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setattr("shiftbench.site.create_app", lambda: "the-app")
    monkeypatch.setitem(
        __import__("sys").modules,
        "uvicorn",
        type(
            "M",
            (),
            {"run": staticmethod(lambda app, host, port: calls.update(app=app, host=host, port=port))},
        ),
    )
    cli.main(["serve", "--port", "9099", "--host", "0.0.0.0"])
    assert calls == {"app": "the-app", "host": "0.0.0.0", "port": 9099}
