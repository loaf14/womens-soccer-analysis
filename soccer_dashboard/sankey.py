"""Categorical shot associations rendered as a Plotly Sankey diagram."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd
import plotly.graph_objects as go

BODY_PART_RARE_SHARE = 0.01
PLAY_PATTERN_RARE_SHARE = 0.01

_PLAY_PATTERN_ORDER = (
    "From Corner",
    "From Counter",
    "From Free Kick",
    "From Goal Kick",
    "From Keeper",
    "From Kick Off",
    "From Throw In",
    "Other",
    "Regular Play",
    "Unknown",
)
_STAGES = (
    ("shot_location", "Shot location"),
    ("play_pattern", "Play pattern"),
    ("body_part", "Body part"),
    ("outcome", "Outcome"),
)
_GOALS_ONLY_STAGES = _STAGES[:-1]
_STAGE_COLORS = {
    "play_pattern": "#62b88a",
    "shot_location": "#78a6d8",
    "body_part": "#e0ad68",
    "outcome": "#d78273",
}
_LINK_COLORS = {
    "play_pattern": "rgba(98,184,138,0.24)",
    "shot_location": "rgba(120,166,216,0.24)",
    "body_part": "rgba(224,173,104,0.24)",
}
_LAYOUT_HEIGHT = 500
_LAYOUT_MARGIN = {"l": 12, "r": 12, "t": 20, "b": 12}
_SANKEY_NODE_PAD = 12
_SANKEY_NODE_X_MARGIN = 0.02
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
    """Return shared stage category orders for the selected display mode."""
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
    play_patterns = set(_PLAY_PATTERN_ORDER)
    for shots in league_shots.values():
        play_patterns.update(shots["play_pattern"].map(_category_label))

    orders = {
        column: _stage_labels(categories, column)
        for column, _ in stages
        if column != "play_pattern"
    }
    orders["play_pattern"] = [
        label for label in _PLAY_PATTERN_ORDER if label in play_patterns
    ] + sorted(play_patterns.difference(_PLAY_PATTERN_ORDER), key=str.casefold)
    return orders


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
    node_x: list[float] = []
    node_y: list[float] = []
    node_index: dict[tuple[str, str], int] = {}
    source_totals: dict[tuple[str, str], int] = {}

    labels_by_stage: dict[str, list[str]] = {}
    for column, _ in stages:
        labels = _stage_labels(categories, column)
        if category_orders is not None and column in category_orders:
            shared_order = list(category_orders[column])
            labels = [label for label in shared_order if label in labels] + [
                label for label in labels if label not in shared_order
            ]
        elif column == "play_pattern":
            labels = [label for label in _PLAY_PATTERN_ORDER if label in labels] + [
                label for label in labels if label not in _PLAY_PATTERN_ORDER
            ]
        labels_by_stage[column] = labels

    total_shots = len(categories)
    plot_height = (
        _LAYOUT_HEIGHT - _LAYOUT_MARGIN["t"] - _LAYOUT_MARGIN["b"]
    )
    max_stage_nodes = max(
        (len(labels) for labels in labels_by_stage.values()), default=0
    )
    node_scale = (
        (plot_height - _SANKEY_NODE_PAD * (max_stage_nodes - 1)) / total_shots
        if total_shots and max_stage_nodes
        else 0
    )

    for stage_index, (column, stage_label) in enumerate(stages):
        labels = labels_by_stage[column]
        color = _STAGE_COLORS[column]
        stage_node_height = (
            sum(int(categories[column].eq(label).sum()) for label in labels)
            * node_scale
            + _SANKEY_NODE_PAD * (len(labels) - 1)
            if labels
            else 0
        )
        current_y = (plot_height - stage_node_height) / 2
        stage_x = _SANKEY_NODE_X_MARGIN + (
            (1 - 2 * _SANKEY_NODE_X_MARGIN)
            * stage_index
            / max(len(stages) - 1, 1)
        )
        for label in labels:
            node_height = int(categories[column].eq(label).sum()) * node_scale
            node_index[(column, label)] = len(node_labels)
            node_labels.append(label)
            node_stages.append(stage_label)
            node_colors.append(color)
            node_x.append(stage_x)
            node_y.append((current_y + node_height / 2) / plot_height)
            source_totals[(column, label)] = int(categories[column].eq(label).sum())
            current_y += node_height + _SANKEY_NODE_PAD

    link_sources: list[int] = []
    link_targets: list[int] = []
    link_values: list[int] = []
    link_customdata: list[list[str | int | float]] = []
    link_colors: list[str] = []

    for (source_column, _), (target_column, _) in zip(
        stages[:-1], stages[1:], strict=True
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
            link_colors.append(_LINK_COLORS[source_column])

    figure = go.Figure(
        go.Sankey(
            uid=f"shot-characteristics-sankey-{mode}",
            arrangement="fixed",
            node={
                "label": node_labels,
                "customdata": node_stages,
                "color": node_colors,
                "x": node_x,
                "y": node_y,
                "pad": _SANKEY_NODE_PAD,
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
        margin=_LAYOUT_MARGIN,
        height=_LAYOUT_HEIGHT,
        autosize=True,
        uirevision=f"shot-characteristics-sankey-{mode}",
    )
    return figure
