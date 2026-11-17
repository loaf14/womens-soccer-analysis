# Women's Soccer Analytics Dashboard

A Python dashboard project for exploring women's soccer data with Pandas, Plotly,
and Dash. Match and event data are retrieved from StatsBomb Open Data through
`statsbombpy`; no credentials are required for the open-data competitions.

## Setup

Python 3.10 or newer is required.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Prepare data

Build a local Parquet cache of NWSL 2023 and FA Women's Super League 2023/24 shot
events:

```powershell
python -m soccer_dashboard.build_data
```

The command downloads each configured season from StatsBomb Open Data once and
writes `data/shots.parquet`. To choose another output location or verify the
historical combined count:

```powershell
python -m soccer_dashboard.build_data --output data/shots.parquet --expected-total 7059
```

The preserved notebook records a historical baseline of 7,059 shots: 3,530
NWSL shots (2023) and 3,529 FA Women's Super League shots (2023/24). This is a
baseline, not a guarantee that the upstream open-data source will never change.
Use `--expected-total 7059` to fail the build if the live source no longer
matches it.

The source is StatsBomb Open Data, retrieved via `statsbombpy`; data availability
is determined by StatsBomb. The StatsBomb Public Data User Agreement governs use:
Section 1.2.1 prohibits redistribution of the underlying data, and Section 1.4
requires the StatsBomb logo on published analysis. Please credit
[StatsBomb Open Data](https://github.com/hudl/open-data) and retain the official
logo in published dashboards. The prepared Parquet file is gitignored and is
never placed in Dash's public `assets/` directory.

## Run the dashboard

```powershell
python -m soccer_dashboard.app
```

The app loads the local Parquet cache rather than making network requests during
startup and displays the total shots across both seasons. The existing
[`exploration.ipynb`](./exploration.ipynb) is retained unchanged.

## Deploy on Render

Create a **Web Service** from this repository, select the `feature/render-deployment`
branch (or `main` after this change is reviewed and merged), and use:

- **Build command:** `pip install . && python -m soccer_dashboard.build_data --expected-total 7059`
- **Start command:** `gunicorn "soccer_dashboard.app:create_server()" --bind 0.0.0.0:$PORT --workers 1 --threads 4`
- **Environment variables:** No custom variables are required. Render supplies
  `PORT`; `.python-version` selects Python 3.12.

The build command fetches the configured seasons directly from StatsBomb and
creates `data/shots.parquet` inside the service build. The start command loads
that prepared cache through the Gunicorn application factory; the application
does not contact StatsBomb during startup. The count check intentionally fails
the build if the upstream total changes from the current 7,059-shot baseline,
so any updated data can be reviewed before deployment. The cache is rebuilt on
each deploy, not each process start. It is not a public Dash asset or download
route; Render's free service does not provide persistent runtime storage.

### Data exposure and licensing note

The Parquet file is not tracked by Git and the app has no route that serves it
for download. Shot maps aggregate observations into 4-by-4-coordinate-unit bins
before constructing Plotly figures: **Binned shots** shows goal and non-goal
counts per bin, **Heatmap** shows total counts per bin, and **Hybrid** combines
the total-count heatmap with aggregated goal counts. No shot IDs, player or team
names, event-level locations, or event metadata are added to these traces.
Nevertheless, even spatially aggregated counts may be governed by the User
Agreement. Confirm that this visualization is permitted before making the
service public; if not, obtain written permission or remove the affected data.

Render's free web services can spin down when idle, so the first request may
take longer. The build re-fetches the source data on each new deploy; an upstream
availability change or a count change will make that build fail rather than
silently deploying an unexpected dataset.

## Tests

```powershell
python -m pytest
```
