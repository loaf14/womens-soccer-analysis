"""League shot-conversion summaries and comparison charts."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.graph_objects as go

_COMPETITIONS = {
    "NWSL": "NWSL",
    "FA Women's Super League": "WSL",
}
_LEAGUES = ("NWSL", "WSL")
_LEAGUE_COLORS = {"NWSL": "#e7a45e", "WSL": "#8db4ff"}
_ASSIST_STATUSES = ("Assisted", "Unassisted")
_PATTERN_COLUMNS = {"competition_name", "play_pattern", "shot_outcome"}
_ASSIST_COLUMNS = {
    "competition_name",
    "shot_key_pass_id",
    "assisted",
    "shot_outcome",
}


def _validated_league_rows(
    shots: pd.DataFrame, required_columns: set[str], purpose: str
) -> pd.DataFrame:
    missing_columns = required_columns.difference(shots.columns)
    if missing_columns:
        raise ValueError(
            f"Shot data is missing {purpose} columns: "
            + ", ".join(sorted(missing_columns))
        )

    if shots["competition_name"].isna().any():
        raise ValueError("Shot data contains missing competition names.")

    unknown_competitions = set(shots["competition_name"]).difference(_COMPETITIONS)
    if unknown_competitions:
        raise ValueError(
            "Unsupported competition(s) for conversion comparison: "
            + ", ".join(sorted(unknown_competitions))
        )

    return shots.assign(league=shots["competition_name"].map(_COMPETITIONS))


def _complete_conversion_summary(
    rows: pd.DataFrame, category_column: str, categories: Sequence[str]
) -> pd.DataFrame:
    observed = (
        rows.assign(is_goal=rows["shot_outcome"].eq("Goal"))
        .groupby(["league", category_column], observed=True)
        .agg(shots=("is_goal", "size"), goals=("is_goal", "sum"))
        .reset_index()
    )
    complete = pd.DataFrame(
        [
            (league, category)
            for league in _LEAGUES
            for category in categories
        ],
        columns=["league", category_column],
    )
    summary = complete.merge(observed, on=["league", category_column], how="left")
    summary["shots"] = summary["shots"].fillna(0).astype(int)
    summary["goals"] = summary["goals"].fillna(0).astype(int)
    summary["conversion_rate"] = (
        summary["goals"]
        .div(summary["shots"])
        .mul(100)
        .where(summary["shots"] > 0)
    )
    return summary


def _play_pattern_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    label = str(value).strip()
    return label or "Unknown"


def prepare_play_pattern_conversion_data(shots: pd.DataFrame) -> pd.DataFrame:
    """Return conversion counts for each league and observed play pattern.

    Missing or blank play patterns are represented as ``Unknown``. A league /
    pattern group with no shots has a NaN conversion rate, not a fabricated zero.
    """
    rows = _validated_league_rows(
        shots, _PATTERN_COLUMNS, "play-pattern conversion"
    )
    rows = rows.assign(play_pattern=rows["play_pattern"].map(_play_pattern_label))
    combined_counts = rows["play_pattern"].value_counts()
    pattern_order = sorted(
        combined_counts.index,
        key=lambda pattern: (-combined_counts[pattern], pattern.casefold()),
    )
    summary = _complete_conversion_summary(rows, "play_pattern", pattern_order)
    summary["combined_shots"] = summary.groupby("play_pattern")["shots"].transform(
        "sum"
    )
    order = {pattern: index for index, pattern in enumerate(pattern_order)}
    league_order = {league: index for index, league in enumerate(_LEAGUES)}
    summary["_pattern_order"] = summary["play_pattern"].map(order)
    summary["_league_order"] = summary["league"].map(league_order)
    return (
        summary.sort_values(["_pattern_order", "_league_order"])
        .drop(columns=["_pattern_order", "_league_order"])
        .reset_index(drop=True)
    )


def prepare_assist_conversion_data(shots: pd.DataFrame) -> pd.DataFrame:
    """Return league/assist conversion counts and each category's shot share.

    Assisted status must be a Boolean column and agree with whether a key-pass
    ID is recorded. Missing shots in a category have no conversion rate or share.
    """
    rows = _validated_league_rows(shots, _ASSIST_COLUMNS, "assist conversion")
    if not rows.empty and not pd.api.types.is_bool_dtype(rows["assisted"].dtype):
        raise ValueError("Shot data 'assisted' column must contain Boolean values.")
    if rows["assisted"].isna().any():
        raise ValueError("Shot data contains missing assisted values.")

    expected_assisted = rows["shot_key_pass_id"].notna()
    if not rows["assisted"].eq(expected_assisted).all():
        raise ValueError(
            "Shot data 'assisted' values must match shot_key_pass_id presence."
        )

    rows = rows.assign(
        assist_status=rows["assisted"].map(
            {True: "Assisted", False: "Unassisted"}
        )
    )
    summary = _complete_conversion_summary(rows, "assist_status", _ASSIST_STATUSES)
    league_totals = rows.groupby("league").size()
    summary["league_shot_share"] = summary.apply(
        lambda row: (
            row["shots"] / league_totals[row["league"]] * 100
            if league_totals.get(row["league"], 0) > 0
            else float("nan")
        ),
        axis=1,
    )
    return summary


def create_play_pattern_conversion_figure(shots: pd.DataFrame) -> go.Figure:
    """Build a league dumbbell chart for conversion by play pattern."""
    summary = prepare_play_pattern_conversion_data(shots)
    patterns = summary["play_pattern"].drop_duplicates().tolist()
    figure = go.Figure()

    for pattern in patterns:
        group = summary.loc[summary["play_pattern"].eq(pattern)].set_index("league")
        if group["conversion_rate"].notna().all():
            figure.add_trace(
                go.Scatter(
                    x=[group.loc[league, "conversion_rate"] for league in _LEAGUES],
                    y=[pattern, pattern],
                    mode="lines",
                    line={"color": "#64736a", "width": 1.5},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{pattern} connector",
                )
            )

    for league in _LEAGUES:
        league_rows = summary.loc[
            summary["league"].eq(league) & summary["conversion_rate"].notna()
        ]
        other_league = "WSL" if league == "NWSL" else "NWSL"
        positions = []
        for row in league_rows.itertuples(index=False):
            other = summary.loc[
                summary["league"].eq(other_league)
                & summary["play_pattern"].eq(row.play_pattern),
                "conversion_rate",
            ].iloc[0]
            positions.append(
                "top center"
                if pd.isna(other) or league == "NWSL"
                else "bottom center"
            )

        figure.add_trace(
            go.Scatter(
                x=league_rows["conversion_rate"],
                y=league_rows["play_pattern"],
                mode="markers+text",
                name=league,
                text=[f"{rate:.1f}%" for rate in league_rows["conversion_rate"]],
                textposition=positions,
                textfont={"color": _LEAGUE_COLORS[league], "size": 10},
                marker={
                    "color": _LEAGUE_COLORS[league],
                    "size": 10,
                    "line": {"color": "#111a17", "width": 1.2},
                },
                customdata=league_rows[
                    ["shots", "goals", "conversion_rate"]
                ].to_numpy(),
                hovertemplate=(
                    f"League: {league}<br>"
                    "Play pattern: %{y}<br>"
                    "Shots: %{customdata[0]:,.0f}<br>"
                    "Goals: %{customdata[1]:,.0f}<br>"
                    "Conversion rate: %{customdata[2]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    maximum_value = summary["conversion_rate"].max(skipna=True)
    maximum_rate = float(maximum_value) if pd.notna(maximum_value) else 0
    axis_maximum = min(100, max(10, ((maximum_rate + 4) // 5) * 5 + 5))
    height = max(390, 160 + 31 * len(patterns))
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#111a17",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 12, "r": 18, "t": 42, "b": 52},
        height=height,
        autosize=True,
        xaxis={
            "title": "Goal conversion rate",
            "range": [0, axis_maximum],
            "ticksuffix": "%",
            "rangemode": "tozero",
            "gridcolor": "rgba(154,168,158,0.14)",
            "zeroline": False,
        },
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(patterns)),
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
        hoverlabel={"bgcolor": "#17231e", "font": {"color": "#f0f3ed"}},
    )
    return figure


def create_assist_conversion_figure(shots: pd.DataFrame) -> go.Figure:
    """Build a grouped conversion-rate chart for recorded assist status."""
    summary = prepare_assist_conversion_data(shots)
    figure = go.Figure()

    for league in _LEAGUES:
        league_rows = summary.loc[summary["league"].eq(league)]
        labels = [
            f"{rate:.1f}%" if pd.notna(rate) else ""
            for rate in league_rows["conversion_rate"]
        ]
        figure.add_trace(
            go.Bar(
                x=league_rows["assist_status"],
                y=league_rows["conversion_rate"],
                name=league,
                marker_color=_LEAGUE_COLORS[league],
                text=labels,
                textposition="outside",
                cliponaxis=False,
                customdata=league_rows[
                    [
                        "shots",
                        "goals",
                        "conversion_rate",
                        "league_shot_share",
                    ]
                ].to_numpy(),
                hovertemplate=(
                    f"League: {league}<br>"
                    "Assist status: %{x}<br>"
                    "Shots: %{customdata[0]:,.0f}<br>"
                    "Goals: %{customdata[1]:,.0f}<br>"
                    "Conversion rate: %{customdata[2]:.1f}%<br>"
                    "Share of league shots: %{customdata[3]:.1f}%"
                    "<extra></extra>"
                ),
            )
        )

    maximum_value = summary["conversion_rate"].max(skipna=True)
    maximum_rate = float(maximum_value) if pd.notna(maximum_value) else 0
    axis_maximum = min(100, max(10, ((maximum_rate + 4) // 5) * 5 + 5))
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#111a17",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 42, "r": 18, "t": 42, "b": 48},
        height=440,
        autosize=True,
        barmode="group",
        bargap=0.35,
        bargroupgap=0.12,
        xaxis={
            "title": None,
            "categoryorder": "array",
            "categoryarray": list(_ASSIST_STATUSES),
        },
        yaxis={
            "title": "Goal conversion rate",
            "range": [0, axis_maximum],
            "ticksuffix": "%",
            "rangemode": "tozero",
            "gridcolor": "rgba(154,168,158,0.14)",
            "zeroline": False,
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        hoverlabel={"bgcolor": "#17231e", "font": {"color": "#f0f3ed"}},
    )
    return figure
