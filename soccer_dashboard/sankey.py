"""Categorical shot associations rendered as a Plotly Sankey diagram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

BODY_PART_RARE_SHARE = 0.01
PLAY_PATTERN_RARE_SHARE = 0.01

_STAGES = (
    ("play_pattern", "Play pattern"),
    ("shot_location", "Shot location"),
    ("body_part", "Body part"),
    ("outcome", "Outcome"),
)
_GOALS_ONLY_STAGES = _STAGES[:-1]
_STAGE_COLORS = ("#62b88a", "#78a6d8", "#e0ad68", "#d78273")
_LINK_COLORS = (
    "rgba(98,184,138,0.24)",
    "rgba(120,166,216,0.24)",
    "rgba(224,173,104,0.24)",
)
_REQUIRED_COLUMNS = {"play_pattern", "in_box", "shot_body_part", "shot_outcome"}


def _category_label(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    label = str(value).strip()
    return label or "Unknown"


def _outcome_category(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    outcome = str(value).strip()
    if not outcome:
        return "Unknown"
    return "Goal" if outcome == "Goal" else "Non-goal"


def _prepare_categories(shots: pd.DataFrame) -> pd.DataFrame:
    missing_columns = _REQUIRED_COLUMNS.difference(shots.columns)
    if missing_columns:
        raise ValueError(
            "Shot data is missing Sankey columns: "
            + ", ".join(sorted(missing_columns))
        )

    categories = pd.DataFrame(index=shots.index)
    play_patterns = shots["play_pattern"].map(_category_label)
    play_pattern_counts = play_patterns.value_counts()
    rare_patterns = (
        {
            pattern
            for pattern, count in play_pattern_counts.items()
            if pattern not in {"Other", "Unknown"}
            and count / len(shots) < PLAY_PATTERN_RARE_SHARE
        }
        if not shots.empty
        else set()
    )
    categories["play_pattern"] = play_patterns.mask(
        play_patterns.isin(rare_patterns), "Other"
    )
    categories["shot_location"] = shots["in_box"].map(
        lambda value: (
            "Unknown"
            if pd.isna(value)
            else "Inside box"
            if bool(value)
            else "Outside box"
        )
    )
    body_parts = shots["shot_body_part"].map(_category_label)
    if not shots.empty:
        minimum_share = BODY_PART_RARE_SHARE
        body_part_counts = body_parts.value_counts()
        rare_parts = {
            part
            for part, count in body_part_counts.items()
            if part not in {"Other", "Unknown"}
            and count / len(shots) < minimum_share
        }
        body_parts = body_parts.mask(body_parts.isin(rare_parts), "Other")
    categories["body_part"] = body_parts
    categories["outcome"] = shots["shot_outcome"].map(_outcome_category)
    return categories


def _stage_labels(categories: pd.DataFrame, column: str) -> list[str]:
    counts = categories[column].value_counts()
    if column == "outcome":
        preferred = ["Goal", "Non-goal", "Unknown"]
        return [label for label in preferred if label in counts.index]
    if column == "shot_location":
        preferred = ["Inside box", "Outside box", "Unknown"]
        return [label for label in preferred if label in counts.index]
    return sorted(counts.index, key=lambda label: (-counts[label], label.casefold()))


def sankey_category_orders(
    league_shots: Mapping[str, pd.DataFrame], mode: str = "all"
) -> dict[str, list[str]]:
    """Return mode-specific stage category orders shared across league figures."""
    if mode not in {"all", "goals"}:
        raise ValueError("Sankey mode must be 'all' or 'goals'.")

    stages = _GOALS_ONLY_STAGES if mode == "goals" else _STAGES
    category_frames = []
    for shots in league_shots.values():
        selected_shots = (
            shots.loc[shots["shot_outcome"].eq("Goal")]
            if mode == "goals"
            else shots
        )
        category_frames.append(_prepare_categories(selected_shots))

    columns = [column for column, _ in stages]
    categories = (
        pd.concat(category_frames, ignore_index=True)[columns]
        if category_frames
        else pd.DataFrame(columns=columns)
    )
    return {
        column: _stage_labels(categories, column)
        for column, _ in stages
    }


def create_sankey_figure(
    shots: pd.DataFrame,
    mode: str = "all",
    *,
    category_orders: Mapping[str, Sequence[str]] | None = None,
) -> go.Figure:
    """Create a shot-count Sankey for all shots or goals only.

    Play-pattern and body-part categories under 1% of the selected observations
    are grouped as ``Other``. Link percentages use the selected-mode total
    leaving the source category as their denominator; they are not conversion
    rates. Missing or blank values are shown as ``Unknown``. In goals-only
    mode, only observations with the literal ``Goal`` outcome are included and
    the Outcome stage is omitted.
    """
    if mode not in {"all", "goals"}:
        raise ValueError("Sankey mode must be 'all' or 'goals'.")
    selected_shots = (
        shots.loc[shots["shot_outcome"].eq("Goal")]
        if mode == "goals"
        else shots
    )
    categories = _prepare_categories(selected_shots)
    stages = _GOALS_ONLY_STAGES if mode == "goals" else _STAGES
    node_labels: list[str] = []
    node_stages: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[str, str], int] = {}
    source_totals: dict[tuple[str, str], int] = {}

    for (column, stage_label), color in zip(
        stages, _STAGE_COLORS[: len(stages)], strict=True
    ):
        labels = _stage_labels(categories, column)
        if category_orders is not None and column in category_orders:
            shared_order = list(category_orders[column])
            labels = [label for label in shared_order if label in labels] + [
                label for label in labels if label not in shared_order
            ]
        for label in labels:
            node_index[(column, label)] = len(node_labels)
            node_labels.append(label)
            node_stages.append(stage_label)
            node_colors.append(color)
            source_totals[(column, label)] = int(categories[column].eq(label).sum())

    link_sources: list[int] = []
    link_targets: list[int] = []
    link_values: list[int] = []
    link_customdata: list[list[str | int | float]] = []
    link_colors: list[str] = []

    for stage_index, ((source_column, _), (target_column, _)) in enumerate(
        zip(stages[:-1], stages[1:], strict=True)
    ):
        counts = (
            categories.groupby([source_column, target_column], sort=True)
            .size()
            .rename("count")
            .reset_index()
        )
        for source_label, target_label, count_value in counts.itertuples(
            index=False, name=None
        ):
            count = int(count_value)
            source_total = source_totals[(source_column, source_label)]
            link_sources.append(node_index[(source_column, source_label)])
            link_targets.append(node_index[(target_column, target_label)])
            link_values.append(count)
            link_customdata.append(
                [
                    source_label,
                    target_label,
                    source_total,
                    count / source_total * 100,
                ]
            )
            link_colors.append(_LINK_COLORS[stage_index])

    figure = go.Figure(
        go.Sankey(
            uid=f"shot-characteristics-sankey-{mode}",
            arrangement="snap",
            node={
                "label": node_labels,
                "customdata": node_stages,
                "color": node_colors,
                "pad": 12,
                "thickness": 12,
                "line": {"color": "rgba(220,230,221,0.26)", "width": 0.6},
                "hovertemplate": (
                    "Stage: %{customdata}<br>Category: %{label}<extra></extra>"
                ),
            },
            link={
                "source": link_sources,
                "target": link_targets,
                "value": link_values,
                "color": link_colors,
                "customdata": link_customdata,
                "hovertemplate": (
                    "Source: %{customdata[0]}<br>"
                    "Destination: %{customdata[1]}<br>"
                    "Shots: %{value:,.0f}<br>"
                    "Share of source category: %{customdata[3]:.1f}%<br>"
                    "Source category total: %{customdata[2]:,.0f} shots"
                    "<extra></extra>"
                ),
            },
        )
    )
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#111a17",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed", "size": 12},
        margin={"l": 12, "r": 12, "t": 20, "b": 12},
        height=500,
        autosize=True,
        uirevision=f"shot-characteristics-sankey-{mode}",
    )
    return figure
