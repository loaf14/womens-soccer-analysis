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

## Data integration

The `soccer_dashboard.data.load_shots` function retrieves competition metadata,
matches, and shot events for the requested StatsBomb competition and season. It
returns one Pandas DataFrame with shot details, competition and season names,
and an `in_box` indicator based on StatsBomb's 120-by-80 pitch coordinates.

```python
from soccer_dashboard.data import load_shots

shots = load_shots(competition_id=49, season_id=107)
```

The function uses StatsBomb Open Data accessed via `statsbombpy`. Data
availability is determined by StatsBomb; the dashboard does not bundle or
fabricate match data. The existing [`exploration.ipynb`](./exploration.ipynb)
is retained as the original exploratory notebook.

## Tests

```powershell
python -m pytest
```
