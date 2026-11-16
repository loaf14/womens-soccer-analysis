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
is determined by StatsBomb. Please credit [StatsBomb Open
Data](https://github.com/statsbomb/open-data) when using the data. Generated
Parquet data is local, gitignored, and is not included in the repository. If the
cache is missing, run the build command above; the dashboard reports the
regeneration command in its error message.

## Run the dashboard

```powershell
python -m soccer_dashboard.app
```

The app loads the local Parquet cache rather than making network requests during
startup and displays the total shots across both seasons. The existing
[`exploration.ipynb`](./exploration.ipynb) is retained unchanged.

## Tests

```powershell
python -m pytest
```
