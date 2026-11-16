"""League and location shot-conversion comparison."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

_COMPETITIONS = {
    "NWSL": "NWSL",
    "FA Women's Super League": "WSL",
}
_LOCATIONS = ("Inside box", "Outside box")
_LEAGUES = ("NWSL", "WSL")
_LEAGUE_COLORS = {"NWSL": "#e7a45e", "WSL": "#8db4ff"}
_REQUIRED_COLUMNS = {"competition_name", "in_box", "shot_outcome"}


def prepare_conversion_data(shots: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Return league/location counts and the number of shots without location.

    Conversion rates are undefined for groups with no shots and are represented
    by ``NaN`` rather than zero.
    """
    missing_columns = _REQUIRED_COLUMNS.difference(shots.columns)
    if missing_columns:
        raise ValueError(
            "Shot data is missing conversion columns: "
            + ", ".join(sorted(missing_columns))
        )

    unknown_competitions = set(shots["competition_name"].dropna()).difference(
        _COMPETITIONS
    )
    if unknown_competitions:
        raise ValueError(
            "Unsupported competition(s) for conversion comparison: "
            + ", ".join(sorted(unknown_competitions))
        )

    rows = shots.assign(
        league=shots["competition_name"].map(_COMPETITIONS),
        location=shots["in_box"].map(
            {True: "Inside box", False: "Outside box"}
        ),
    )
    missing_location_count = int(rows["location"].isna().sum())
    located = rows.loc[rows["location"].notna()]
    grouped = (
        located.assign(is_goal=located["shot_outcome"].eq("Goal"))
        .groupby(["league", "location"], observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [_LEAGUES, _LOCATIONS], names=["league", "location"]
    ).to_frame(index=False)
    summary = complete.merge(grouped, on=["league", "location"], how="left")
    summary["shots"] = summary["shots"].fillna(0).astype(int)
    summary["goals"] = summary["goals"].fillna(0).astype(int)
    summary["conversion_rate"] = (
        summary["goals"].div(summary["shots"]).mul(100).where(summary["shots"] > 0)
    )
    return summary, missing_location_count


def create_conversion_figure(shots: pd.DataFrame) -> go.Figure:
    """Build a responsive dumbbell chart with counts and rates in hover."""
    summary, _ = prepare_conversion_data(shots)
    figure = go.Figure()

    for location in _LOCATIONS:
        group = summary.loc[summary["location"].eq(location)].set_index("league")
        available = group["conversion_rate"].dropna()
        if len(available) == len(_LEAGUES):
            figure.add_trace(
                go.Scatter(
                    x=[group.loc[league, "conversion_rate"] for league in _LEAGUES],
                    y=[location, location],
                    mode="lines",
                    line={"color": "#64736a", "width": 2},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{location} connector",
                )
            )

    for league in _LEAGUES:
        league_rows = summary.loc[summary["league"].eq(league)]
        valid = league_rows.loc[league_rows["conversion_rate"].notna()]
        labels: list[str] = []
        positions: list[str] = []
        for row in valid.itertuples(index=False):
            rate = float(row.conversion_rate)
            other = summary.loc[
                summary["location"].eq(row.location)
                & summary["league"].ne(league),
                "conversion_rate",
            ].dropna()
            other_rate = float(other.iloc[0]) if len(other) else None
            can_label = other_rate is None or (
                rate != other_rate
                and (
                    min(rate, other_rate) >= 3
                    or abs(rate - other_rate) >= 3
                )
            )
            labels.append(f"{rate:.1f}%" if can_label else "")
            positions.append(
                "middle left"
                if other_rate is not None and rate < other_rate and rate >= 3
                else "middle right"
            )

        figure.add_trace(
            go.Scatter(
                x=valid["conversion_rate"],
                y=valid["location"],
                mode="markers+text",
                name=league,
                text=labels,
                textposition=positions,
                textfont={"color": _LEAGUE_COLORS[league], "size": 11},
                marker={
                    "color": _LEAGUE_COLORS[league],
                    "size": 13,
                    "line": {"color": "#111a17", "width": 1.5},
                },
                customdata=valid[["shots", "goals", "conversion_rate"]].to_numpy(),
                hovertemplate=(
                    f"League: {league}<br>"
                    "Location: %{y}<br>"
                    "Shots: %{customdata[0]:,.0f}<br>"
                    "Goals: %{customdata[1]:,.0f}<br>"
                    "Conversion rate: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    annotations = []
    for row in summary.loc[summary["shots"].eq(0)].itertuples(index=False):
        annotations.append(
            {
                "x": 0,
                "y": row.location,
                "xref": "x",
                "yref": "y",
                "text": f"{row.league}: no shots",
                "showarrow": False,
                "xanchor": "left",
                "yshift": 14 if row.league == "NWSL" else -14,
                "font": {"color": "#9aa89e", "size": 10},
            }
        )

    maximum_value = summary["conversion_rate"].max(skipna=True)
    maximum_rate = float(maximum_value) if pd.notna(maximum_value) else 0
    axis_maximum = min(100, max(10, ((maximum_rate + 4) // 5) * 5 + 5))
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#111a17",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 18, "r": 30, "t": 24, "b": 48},
        height=260,
        autosize=True,
        xaxis={
            "title": "Shot conversion rate",
            "range": [0, axis_maximum],
            "ticksuffix": "%",
            "rangemode": "tozero",
            "gridcolor": "rgba(154,168,158,0.14)",
            "zeroline": False,
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(_LOCATIONS)),
            "title": None,
            "showgrid": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        annotations=annotations,
        hoverlabel={"bgcolor": "#17231e", "font": {"color": "#f0f3ed"}},
    )
    return figure
