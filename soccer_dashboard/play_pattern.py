"""Compare shot distributions by attacking play pattern across leagues."""

from __future__ import annotations

from html import escape
import math

import pandas as pd
import plotly.graph_objects as go

_COMPETITIONS = {
    "NWSL": "NWSL",
    "FA Women's Super League": "WSL",
}
_LEAGUES = ("NWSL", "WSL")
_LEAGUE_COLORS = {"NWSL": "#e7a45e", "WSL": "#8db4ff"}
_REQUIRED_COLUMNS = {"competition_name", "play_pattern"}
_MISSING_PATTERN = "Missing play pattern"


def prepare_play_pattern_data(shots: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return each league's shot counts and percentage for every pattern.

    The denominator is all shot records in a league. Missing and blank play
    patterns remain in the distribution as a distinct category.
    """
    missing_columns = _REQUIRED_COLUMNS.difference(shots.columns)
    if missing_columns:
        raise ValueError(
            "Shot data is missing play-pattern columns: "
            + ", ".join(sorted(missing_columns))
        )

    competition = shots["competition_name"]
    if competition.isna().any():
        raise ValueError("Shot data contains missing competition names")
    unknown_competitions = set(competition.unique()).difference(_COMPETITIONS)
    if unknown_competitions:
        raise ValueError(
            "Unsupported competition(s) for play-pattern comparison: "
            + ", ".join(sorted(unknown_competitions))
        )

    patterns = shots["play_pattern"].astype("string").str.strip()
    missing_pattern = (patterns.isna() | patterns.eq("")).fillna(False)
    observed_patterns = set(patterns.loc[~missing_pattern].astype(str))
    missing_label = _MISSING_PATTERN
    while missing_label in observed_patterns:
        missing_label += " (not recorded)"

    rows = pd.DataFrame(
        {
            "league": competition.map(_COMPETITIONS),
            "play_pattern": patterns.fillna("").mask(
                missing_pattern, missing_label
            ),
        }
    )
    grouped = (
        rows.groupby(["league", "play_pattern"], observed=True)
        .size()
        .rename("shots")
        .reset_index()
    )
    pattern_totals = grouped.groupby("play_pattern")["shots"].sum()
    ordered_patterns = sorted(
        pattern_totals.index,
        key=lambda pattern: (-int(pattern_totals.loc[pattern]), pattern.casefold()),
    )
    counts = (
        grouped.set_index(["league", "play_pattern"])["shots"].to_dict()
        if not grouped.empty
        else {}
    )
    league_totals = rows.groupby("league").size().to_dict()

    records = []
    for pattern in ordered_patterns:
        for league in _LEAGUES:
            shots_count = int(counts.get((league, pattern), 0))
            total_shots = int(league_totals.get(league, 0))
            records.append(
                {
                    "league": league,
                    "play_pattern": pattern,
                    "shots": shots_count,
                    "total_shots": total_shots,
                    "percentage": (
                        shots_count / total_shots * 100 if total_shots else 0.0
                    ),
                }
            )

    return pd.DataFrame(
        records,
        columns=["league", "play_pattern", "shots", "total_shots", "percentage"],
    ), int(missing_pattern.sum())


def create_play_pattern_figure(shots: pd.DataFrame) -> go.Figure:
    """Build a responsive butterfly chart with league-normalized percentages."""
    summary, _ = prepare_play_pattern_data(shots)
    categories = summary["play_pattern"].drop_duplicates().tolist()
    maximum_percentage = (
        float(summary["percentage"].max()) if not summary.empty else 0.0
    )
    if maximum_percentage <= 30:
        tick_step = 5
    elif maximum_percentage <= 80:
        tick_step = 10
    else:
        tick_step = 20
    axis_maximum = min(
        100, max(tick_step, math.ceil(maximum_percentage / tick_step) * tick_step)
    )
    tick_values = list(range(-axis_maximum, axis_maximum + tick_step, tick_step))

    figure = go.Figure()
    for league in _LEAGUES:
        league_rows = summary.loc[summary["league"].eq(league)]
        direction = -1 if league == "NWSL" else 1
        figure.add_trace(
            go.Bar(
                x=league_rows["percentage"] * direction,
                y=league_rows["play_pattern"],
                orientation="h",
                name=league,
                marker={
                    "color": _LEAGUE_COLORS[league],
                    "line": {"color": "#111a17", "width": 1},
                },
                customdata=league_rows[["shots", "percentage"]].to_numpy(),
                hovertemplate=(
                    f"League: {league}<br>"
                    "Play pattern: %{y}<br>"
                    "Shots: %{customdata[0]:,.0f}<br>"
                    "Share of league shots: %{customdata[1]:.2f}%"
                    "<extra></extra>"
                ),
            )
        )

    annotations = [
        {
            "x": 0,
            "y": pattern,
            "xref": "x",
            "yref": "y",
            "text": escape(pattern),
            "showarrow": False,
            "xanchor": "center",
            "align": "center",
            "width": 150,
            "bgcolor": "#111a17",
            "borderpad": 3,
            "font": {"color": "#f0f3ed", "size": 11},
        }
        for pattern in categories
    ]
    height = max(340, 36 * len(categories) + 128)
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#111a17",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 16, "r": 16, "t": 58, "b": 48},
        height=height,
        autosize=True,
        barmode="overlay",
        bargap=0.28,
        xaxis={
            "title": "Share of each league's shots",
            "range": [-axis_maximum, axis_maximum],
            "tickmode": "array",
            "tickvals": tick_values,
            "ticktext": [f"{abs(value)}%" if value else "" for value in tick_values],
            "gridcolor": "rgba(154,168,158,0.14)",
            "zeroline": False,
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": categories,
            "autorange": "reversed",
            "showticklabels": False,
            "showgrid": False,
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "center",
            "x": 0.5,
        },
        annotations=annotations,
        hoverlabel={"bgcolor": "#17231e", "font": {"color": "#f0f3ed"}},
    )
    return figure
