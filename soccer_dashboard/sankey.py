"""Categorical shot associations rendered as a Plotly Sankey diagram."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BODY_PART_RARE_SHARE = 0.01

_STAGES = (
    ("play_pattern", "Play pattern"),
    ("shot_location", "Shot location"),
    ("body_part", "Body part"),
    ("outcome", "Outcome"),
)
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
    categories["play_pattern"] = shots["play_pattern"].map(_category_label)
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


def create_sankey_figure(shots: pd.DataFrame) -> go.Figure:
    """Create links whose widths are shot counts for adjacent category stages.

    Link hover percentages use the total shots leaving that link's source
    category as their denominator. Missing or blank values are shown as
    ``Unknown``; only the ``Goal`` outcome is classified as a goal, and other
    valid outcomes are grouped as ``Non-goal``.
    """
    categories = _prepare_categories(shots)
    node_labels: list[str] = []
    node_stages: list[str] = []
    node_colors: list[str] = []
    node_index: dict[tuple[str, str], int] = {}
    source_totals: dict[tuple[str, str], int] = {}

    for (column, stage_label), color in zip(_STAGES, _STAGE_COLORS, strict=True):
        for label in _stage_labels(categories, column):
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
        zip(_STAGES[:-1], _STAGES[1:], strict=True)
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
            arrangement="snap",
            node={
                "label": node_labels,
                "customdata": node_stages,
                "color": node_colors,
                "pad": 18,
                "thickness": 18,
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
        height=700,
        autosize=True,
        uirevision="shot-characteristics-sankey",
    )
    return figure
