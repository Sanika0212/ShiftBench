"""The README's results section, which is generated so it cannot drift from results/."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "render_readme.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("render_readme", MODULE)
    assert spec
    assert spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pair(
    detector: str,
    source: str,
    target: str,
    method: str,
    achieved: float,
    alarms: float = 0.3,
    spread: float = float("nan"),
    separability: float = float("nan"),
    coverage: float = float("nan"),
) -> dict[str, Any]:
    return {
        "detector": detector,
        "source": source,
        "target": target,
        "method": method,
        "threshold": 0.5,
        "promised": 0.90,
        "achieved": achieved,
        "achieved_low": achieved - 0.02,
        "achieved_high": achieved + 0.02,
        "false_alarms": alarms,
        "gap": achieved - 0.90,
        "n_source": 300,
        "n_target": 300,
        "weight_spread": spread,
        "domain_separability": separability,
        "group_coverage": coverage,
    }


def results(tmp_path: Path, weighted_spread: float, separability: float) -> Path:
    root = tmp_path / "results"
    root.mkdir()
    rows = []
    for source, target in (("a", "b"), ("b", "a"), ("a", "a"), ("b", "b")):
        for method, achieved in (
            ("split", 0.80),
            ("weighted", 0.82),
            ("mondrian", 0.81),
            ("oracle", 0.91),
        ):
            spread = weighted_spread if method == "weighted" else float("nan")
            sep = separability if method == "weighted" else float("nan")
            covered = 1.0 if method == "mondrian" else float("nan")
            rows.append(
                pair(
                    "fake",
                    source,
                    target,
                    method,
                    achieved,
                    spread=spread,
                    separability=sep,
                    coverage=covered,
                )
            )
    pd.DataFrame(rows).to_parquet(root / "pairs.parquet", index=False)
    pd.DataFrame(
        [
            {
                "detector": "fake",
                "pairs": 2,
                "mean_gap": -0.1,
                "worst_gap": -0.2,
                "promise_kept": 0.5,
                "mean_false_alarms": 0.3,
                "labels_to_recover": float("nan"),
            }
        ]
    ).to_parquet(root / "leaderboard.parquet", index=False)
    pd.DataFrame(
        [
            {
                "detector": "fake",
                "target": "b",
                "labels": k,
                "recall": 0.92,
                "recall_sd": sd,
                "within_tolerance": 0.8,
                "false_alarm_rate": 0.3,
                "draws": 50,
            }
            for k, sd in ((25, 0.09), (400, 0.03))
        ]
    ).to_parquet(root / "budgets.parquet", index=False)
    pd.DataFrame(
        [{"detector": "fake", "source": "a", "target": "b", "psi": 0.2, "mmd": 0.1, "mmd_p_value": 0.01}]
    ).to_parquet(root / "drift.parquet", index=False)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "promised": 0.90,
                "detectors": ["fake"],
                "domains": ["a", "b"],
                "claims": 600,
                "drift_predicts_loss": {
                    "pairs": 2.0,
                    "undercovering_pairs": 1.0,
                    "psi_spearman": -0.2,
                    "psi_spearman_absolute": 0.6,
                    "psi_spearman_undercovering": 0.7,
                    "mmd_spearman": -0.1,
                    "mmd_spearman_absolute": 0.8,
                    "mmd_spearman_undercovering": 0.75,
                },
            }
        )
    )
    return root


def render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spread: float, sep: float) -> str:
    module = load_module()
    monkeypatch.setattr(module, "RESULTS", results(tmp_path, spread, sep))
    return module.body(module.load())


def test_every_part_of_the_section_renders(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = render(tmp_path, monkeypatch, spread=3.0, sep=0.6)
    for heading in ("## Results", "What the repairs are worth", "get the promise back", "warn before"):
        assert heading in text
    assert "| fake | 50% |" in text
    assert "| oracle | 0.910 |" in text


def test_a_null_result_is_not_reported_as_a_degenerate_estimator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Weights that varied and did not help is a different finding from weights that did not vary."""
    text = render(tmp_path, monkeypatch, spread=3.0, sep=0.1)
    assert "real negative result" in text
    assert "never ran" not in text


def test_a_degenerate_estimator_is_not_reported_as_a_null_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = render(tmp_path, monkeypatch, spread=0.01, sep=1.0)
    assert "never ran" in text
    assert "real negative result" not in text


def test_the_drift_section_leads_with_the_pairs_that_broke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = render(tmp_path, monkeypatch, spread=3.0, sep=0.6)
    assert "| PSI | -0.20 | +0.60 | +0.70 |" in text
    assert "direction-blind" in text


def test_an_incomplete_results_directory_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "RESULTS", tmp_path / "nothing")
    with pytest.raises(SystemExit, match="make score"):
        module.load()


def test_rendering_twice_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_module()
    monkeypatch.setattr(module, "RESULTS", results(tmp_path, 3.0, 0.6))
    rendered = module.body(module.load())
    once = module.replace("before\n\nRESULTS_SECTION\n\nafter\n", rendered)
    twice = module.replace(once, rendered)
    assert once == twice
    assert "RESULTS_SECTION" not in once


def test_a_mondrian_row_that_never_ran_says_so(tmp_path: Path) -> None:
    """Zero coverage means every claim took the pooled threshold, which is the split one.

    That is what the method did while its groups were document ids, and the only sign was
    its recall column matching split exactly. The section has to name it either way.
    """
    module = load_module()
    rows = []
    for source, target in (("a", "b"), ("b", "a")):
        for method, achieved in (("split", 0.80), ("weighted", 0.82), ("mondrian", 0.80), ("oracle", 0.91)):
            rows.append(
                pair(
                    "fake",
                    source,
                    target,
                    method,
                    achieved,
                    spread=3.0 if method == "weighted" else float("nan"),
                    separability=0.5 if method == "weighted" else float("nan"),
                    coverage=0.0 if method == "mondrian" else float("nan"),
                )
            )
    text = "\n".join(module.methods({"pairs": pd.DataFrame(rows), "summary": {"promised": 0.90}}))
    assert "0% of target claims" in text
    assert "under another name" in text
