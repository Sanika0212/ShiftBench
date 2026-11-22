"""The leaderboard: the results tables as a small site.

One page, server rendered, reading the parquet files the analysis wrote. It ranks detectors
by whether their promise survives a change of domain, shows the source by target matrix
behind each row, and links to the label-budget curve that says what the repair costs.

    shiftbench serve --port 8090

The same tables load into BigQuery (``infra/bigquery/schema.sql``) and this app runs on
Cloud Run, which is how it would be published. Neither has been applied: there is no GCP
project behind this repository, and the README says so.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

RESULTS = Path("results")

STYLE = """
:root {
  --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --line: #e1e0d9;
  --good: #0ca30c; --bad: #d03b3b; --warn: #fab219; --accent: #2a78d6;
}
@media (prefers-color-scheme: dark) {
  :root { --surface: #1a1a19; --ink: #fff; --ink2: #c3c2b7; --line: #2c2c2a; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 2.5rem 1.25rem 4rem; background: var(--surface); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 68rem; margin: 0 auto; }
h1 { font-size: 1.9rem; margin: 0 0 .4rem; letter-spacing: -0.01em; }
h2 { font-size: 1.15rem; margin: 2.5rem 0 .6rem; }
p.lede { color: var(--ink2); margin: 0 0 2rem; max-width: 46rem; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: .5rem .6rem; border-bottom: 1px solid var(--line); }
th { color: var(--ink2); font-weight: 600; font-size: .82rem; text-transform: uppercase;
  letter-spacing: .04em; }
td.num, th.num { text-align: right; }
.good { color: var(--good); } .bad { color: var(--bad); } .warn { color: var(--warn); }
.note { color: var(--ink2); font-size: .85rem; margin-top: .6rem; }
.matrix td.cell { text-align: right; font-size: .85rem; }
"""


def load_results(root: Path = RESULTS) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("leaderboard", "pairs", "budgets", "drift"):
        path = root / f"{name}.parquet"
        if path.exists():
            out[name] = pd.read_parquet(path)
    return out


def _cell(value: float, promised: float, tolerance: float = 0.02) -> str:
    if value != value:
        return '<td class="cell">n/a</td>'
    klass = "good" if value >= promised - tolerance else ("warn" if value >= promised - 0.1 else "bad")
    return f'<td class="cell {klass}">{value:.2f}</td>'


def render(results: dict[str, pd.DataFrame], promised: float = 0.90) -> str:
    if "leaderboard" not in results:
        return (
            f"<!doctype html><meta charset=utf-8><title>ShiftBench</title><style>{STYLE}</style>"
            "<main><h1>ShiftBench</h1><p class=lede>No results yet. Run "
            "<code>shiftbench analyse</code>.</p></main>"
        )
    board, pairs = results["leaderboard"], results.get("pairs")
    rows = []
    for _, r in board.iterrows():
        # "none tried" rather than "n/a": the budgets were measured and none of them cleared
        # the bar on every target, which is a result and not a missing measurement.
        reached = r["labels_to_recover"] == r["labels_to_recover"]
        labels = int(r["labels_to_recover"]) if reached else "none tried"
        rows.append(
            f"<tr><td>{r['detector']}</td>"
            f"<td class=num>{r['promise_kept']:.0%}</td>"
            f"<td class=num>{r['mean_gap']:+.3f}</td>"
            f"<td class=num>{r['worst_gap']:+.3f}</td>"
            f"<td class=num>{r['mean_false_alarms']:.2f}</td>"
            f"<td class=num>{labels}</td></tr>"
        )

    matrices = ""
    if pairs is not None:
        split = pairs[pairs["method"] == "split"]
        for detector, group in split.groupby("detector"):
            domains = sorted(set(group["source"]) | set(group["target"]))
            head = "".join(f'<th class="num">{d}</th>' for d in domains)
            body = ""
            for source in domains:
                cells = ""
                for target in domains:
                    hit = group[(group["source"] == source) & (group["target"] == target)]
                    cells += _cell(float(hit["achieved"].iloc[0]) if len(hit) else float("nan"), promised)
                body += f"<tr><th>{source}</th>{cells}</tr>"
            matrices += (
                f"<h2>{detector}</h2>"
                f'<table class="matrix"><tr><th>calibrated on</th>{head}</tr>{body}</table>'
                f'<p class="note">Rows are the domain the threshold was fitted on, columns the '
                f"domain it was used in. Each cell is the share of unsupported claims actually "
                f"caught against a promise of {promised:.0%}.</p>"
            )

    return f"""<!doctype html>
<meta charset=utf-8>
<meta name=viewport content="width=device-width, initial-scale=1">
<title>ShiftBench</title>
<style>{STYLE}</style>
<main>
  <h1>ShiftBench</h1>
  <p class="lede">A grounding checker is a score and a threshold. The threshold is fitted
  where there are labels and used where there are none. This leaderboard ranks detectors by
  whether the promise that threshold makes survives the move, and by how many labels it takes
  to get it back.</p>
  <table>
    <tr><th>detector</th><th class="num">promise kept</th><th class="num">mean gap</th>
      <th class="num">worst gap</th><th class="num">false alarms</th>
      <th class="num">labels to recover</th></tr>
    {"".join(rows)}
  </table>
  <p class="note">Promise kept is the share of source and target pairs where the achieved
  recall stayed within two points of what was promised. The gap is achieved minus promised,
  so negative is a broken promise.</p>
  {matrices}
</main>
"""


def create_app(root: Path = RESULTS) -> Any:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="ShiftBench")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return render(load_results(root))

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/{table}")
    def table(table: str) -> JSONResponse:
        results = load_results(root)
        if table not in results:
            return JSONResponse({"error": f"no table {table}"}, status_code=404)
        return JSONResponse(results[table].to_dict("records"))

    return app
