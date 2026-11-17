"""Plotly shot-map figures using StatsBomb's 120-by-80 pitch coordinates."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

ShotMapMode = Literal["binned", "heatmap", "hybrid"]

PITCH_LENGTH = 120
PITCH_WIDTH = 80
HALF_PITCH_START_X = 60
HALF_PITCH_LENGTH = PITCH_LENGTH - HALF_PITCH_START_X
HALF_PITCH_WIDTH = PITCH_WIDTH
DENSITY_BIN_SIZE = 4
SHOT_MAP_X_RANGE = (-30, 30)
SHOT_MAP_Y_RANGE = (35, 63)
SHOT_MAP_VIEW_STATSBOMB_X_RANGE = (95, 120)
SHOT_MAP_VIEW_STATSBOMB_Y_RANGE = (10, 70)
GOAL_COLOR = "#f0ad5e"
SHOT_COLOR = "#a9b8e8"
PITCH_LINE_COLOR = "#88a18f"
HEATMAP_COLORS = [
    [0, "#10231b"],
    [0.3, "#246342"],
    [0.68, "#49bd70"],
    [1, "#c5edaf"],
]

def _pitch_coordinates(location: object) -> tuple[float, float] | None:
    if isinstance(location, (str, bytes)) or not pd.api.types.is_list_like(location):
        return None

    coordinates = list(location)
    if len(coordinates) < 2:
        return None

    x, y = coordinates[:2]
    if (
        not isinstance(x, Real)
        or not isinstance(y, Real)
        or not math.isfinite(float(x))
        or not math.isfinite(float(y))
        or not 0 <= x <= PITCH_LENGTH
        or not 0 <= y <= PITCH_WIDTH
    ):
        return None
    return float(x), float(y)


def prepare_shot_coordinates(shots: pd.DataFrame) -> pd.DataFrame:
    """Copy shots and add transformed map coordinates and coverage flags.

    Map x is the original lateral coordinate centered at zero; map y is the
    original x coordinate offset by 60, so attacks toward x=120 point upward.
    Shot observations inside the focused visible window are at StatsBomb
    x=95..120 and y=10..70; extra axis margin shows the goal beyond x=120.
    """
    if "location" not in shots.columns:
        raise ValueError("Shot data must include a 'location' column")

    prepared = shots.copy().reset_index(drop=True)
    coordinates = [_pitch_coordinates(location) for location in prepared["location"]]
    prepared["statsbomb_x"] = [point[0] if point else math.nan for point in coordinates]
    prepared["statsbomb_y"] = [point[1] if point else math.nan for point in coordinates]
    prepared["map_x"] = prepared["statsbomb_y"] - PITCH_WIDTH / 2
    prepared["map_y"] = prepared["statsbomb_x"] - HALF_PITCH_START_X
    prepared["has_valid_location"] = prepared["statsbomb_x"].notna()
    prepared["in_attacking_half"] = (
        prepared["has_valid_location"]
        & prepared["statsbomb_x"].ge(HALF_PITCH_START_X)
    )
    prepared["in_shot_map_view"] = (
        prepared["has_valid_location"]
        & prepared["statsbomb_x"].between(*SHOT_MAP_VIEW_STATSBOMB_X_RANGE)
        & prepared["statsbomb_y"].between(*SHOT_MAP_VIEW_STATSBOMB_Y_RANGE)
    )
    outcomes = prepared.get(
        "shot_outcome", pd.Series(index=prepared.index, dtype="object")
    )
    prepared["is_goal"] = outcomes.eq("Goal").fillna(False).astype(bool)
    return prepared


def _line_shape(
    name: str, x0: float, y0: float, x1: float, y1: float
) -> dict[str, object]:
    return {
        "type": "line",
        "name": name,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": {"color": PITCH_LINE_COLOR, "width": 1.2},
    }


def _rectangle_shape(
    name: str, x0: float, y0: float, x1: float, y1: float
) -> dict[str, object]:
    return {
        "type": "rect",
        "name": name,
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "line": {"color": PITCH_LINE_COLOR, "width": 1.2},
        "fillcolor": "rgba(0,0,0,0)",
    }


def pitch_shapes() -> list[dict[str, object]]:
    """Return exact half-pitch, box, spot, center-circle, and goal geometry."""
    lateral_extent = HALF_PITCH_WIDTH / 2
    penalty_area_half_width = 22
    six_yard_half_width = 10
    return [
        _rectangle_shape(
            "pitch-boundary", -lateral_extent, 0, lateral_extent, HALF_PITCH_LENGTH
        ),
        _line_shape("halfway-line", -lateral_extent, 0, lateral_extent, 0),
        _rectangle_shape("penalty-area", -penalty_area_half_width, 42, penalty_area_half_width, 60),
        _rectangle_shape("six-yard-box", -six_yard_half_width, 54, six_yard_half_width, 60),
        _line_shape("goal-line", -lateral_extent, 60, lateral_extent, 60),
        {
            "type": "circle",
            "name": "center-circle",
            "x0": -9.15,
            "y0": -9.15,
            "x1": 9.15,
            "y1": 9.15,
            "line": {"color": PITCH_LINE_COLOR, "width": 1.2},
        },
        {
            "type": "circle",
            "name": "penalty-spot",
            "x0": -0.35,
            "y0": 47.65,
            "x1": 0.35,
            "y1": 48.35,
            "line": {"color": PITCH_LINE_COLOR, "width": 1},
            "fillcolor": PITCH_LINE_COLOR,
        },
        _rectangle_shape("goal", -4, 60, 4, 62),
        _line_shape("left-goal-post", -4, 60, -4, 62),
        _line_shape("right-goal-post", 4, 60, 4, 62),
        _line_shape("goal-back", -4, 62, 4, 62),
    ]


def _pitch_arc_traces() -> list[go.Scatter]:
    arc_angles = [
        math.radians(126.87 + (233.13 - 126.87) * step / 40)
        for step in range(41)
    ]
    penalty_x = [10 * math.sin(angle) for angle in arc_angles]
    penalty_y = [48 + 10 * math.cos(angle) for angle in arc_angles]

    left_corner_angles = [math.radians(270 + step * 90 / 20) for step in range(21)]
    right_corner_angles = [math.radians(180 + step * 90 / 20) for step in range(21)]

    def corner_points(
        center_x: float, angles: list[float]
    ) -> tuple[list[float], list[float]]:
        return (
            [center_x + math.cos(angle) for angle in angles],
            [60 + math.sin(angle) for angle in angles],
        )

    left_x, left_y = corner_points(-40, left_corner_angles)
    right_x, right_y = corner_points(40, right_corner_angles)
    return [
        go.Scatter(
            x=penalty_x,
            y=penalty_y,
            mode="lines",
            line={"color": PITCH_LINE_COLOR, "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
            name="Penalty arc",
        ),
        go.Scatter(
            x=left_x + [None] + right_x,
            y=left_y + [None] + right_y,
            mode="lines",
            line={"color": PITCH_LINE_COLOR, "width": 1.2},
            hoverinfo="skip",
            showlegend=False,
            name="Corner arcs",
        ),
    ]


def _binned_shot_counts(shots: pd.DataFrame) -> pd.DataFrame:
    x_start = -HALF_PITCH_WIDTH / 2
    x_bin_count = int(HALF_PITCH_WIDTH / DENSITY_BIN_SIZE)
    y_bin_count = int(HALF_PITCH_LENGTH / DENSITY_BIN_SIZE)
    binned = shots.assign(
        x_bin=((shots["map_x"] - x_start) / DENSITY_BIN_SIZE)
        .astype(int)
        .clip(upper=x_bin_count - 1),
        y_bin=(shots["map_y"] / DENSITY_BIN_SIZE)
        .astype(int)
        .clip(upper=y_bin_count - 1),
    )
    counts = (
        binned.groupby(["x_bin", "y_bin", "is_goal"], sort=True)
        .size()
        .rename("shot_count")
        .reset_index()
    )
    counts["map_x"] = x_start + (counts["x_bin"] + 0.5) * DENSITY_BIN_SIZE
    counts["map_y"] = (counts["y_bin"] + 0.5) * DENSITY_BIN_SIZE
    return counts


def _density_grid(
    shots: pd.DataFrame,
) -> tuple[list[float], list[float], list[list[int]], int]:
    x_start = -HALF_PITCH_WIDTH / 2
    x_bin_count = int(HALF_PITCH_WIDTH / DENSITY_BIN_SIZE)
    y_bin_count = int(HALF_PITCH_LENGTH / DENSITY_BIN_SIZE)
    counts = [[0] * y_bin_count for _ in range(x_bin_count)]
    for row in _binned_shot_counts(shots).itertuples(index=False):
        counts[row.x_bin][row.y_bin] += row.shot_count

    x_centers = [
        x_start + (index + 0.5) * DENSITY_BIN_SIZE
        for index in range(x_bin_count)
    ]
    y_centers = [
        (index + 0.5) * DENSITY_BIN_SIZE for index in range(y_bin_count)
    ]
    z_values = [
        [counts[x_index][y_index] for x_index in range(x_bin_count)]
        for y_index in range(y_bin_count)
    ]
    return x_centers, y_centers, z_values, max(map(max, counts))


def max_shot_density(shots: pd.DataFrame) -> int:
    """Return the largest 4-by-4 attacking-half shot count for a dataset."""
    prepared = prepare_shot_coordinates(shots)
    in_half = prepared.loc[prepared["in_attacking_half"]]
    return _density_grid(in_half)[3]


def _add_density(
    fig: go.Figure, shots: pd.DataFrame, density_max: int | None
) -> None:
    x_centers, y_centers, z_values, actual_max = _density_grid(shots)
    if density_max is not None and density_max < actual_max:
        raise ValueError(
            "density_max cannot be lower than the maximum shot count per bin"
        )
    fig.add_trace(
        go.Heatmap(
            x=x_centers,
            y=y_centers,
            z=z_values,
            colorscale=HEATMAP_COLORS,
            colorbar={
                "title": {
                    "text": "Shot attempts per bin",
                    "side": "right",
                    "font": {"color": "#dce6dd"},
                },
                "tickfont": {"color": "#dce6dd"},
                "outlinewidth": 0,
                "thickness": 14,
                "len": 0.72,
                "bgcolor": "rgba(11,17,16,0.82)",
            },
            zmin=0,
            zmax=max(1, density_max if density_max is not None else actual_max),
            hoverongaps=False,
            hovertemplate=(
                "Shot attempts: %{z}<br>"
                "Lateral position: %{x:.0f}<br>"
                "Distance from halfway: %{y:.0f}"
                "<extra></extra>"
            ),
            name="Shot frequency",
            showlegend=False,
            opacity=0.88,
        )
    )


def _add_shot_markers(
    fig: go.Figure, shots: pd.DataFrame, *, goals_only: bool
) -> None:
    binned_counts = _binned_shot_counts(shots)
    if goals_only:
        groups = [(True, "Goals", GOAL_COLOR, 11)]
    else:
        groups = [
            (False, "Non-goals", SHOT_COLOR, 7),
            (True, "Goals", GOAL_COLOR, 11),
        ]

    for is_goal, label, color, size in groups:
        group = binned_counts.loc[binned_counts["is_goal"].eq(is_goal)]
        fig.add_trace(
            go.Scattergl(
                x=group["map_x"],
                y=group["map_y"],
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "size": [
                        size + min(12, 3 * math.sqrt(count))
                        for count in group["shot_count"]
                    ],
                    "opacity": 0.86 if is_goal else 0.48,
                    "line": {
                        "color": "#f8f2e7" if is_goal else "rgba(0,0,0,0)",
                        "width": 1 if is_goal else 0,
                    },
                },
                customdata=[[int(count)] for count in group["shot_count"]],
                hovertemplate=(
                    f"{label} in 4-by-4-unit bin: "
                    "%{customdata[0]:,.0f}<extra></extra>"
                ),
            )
        )


def create_shot_map(
    shots: pd.DataFrame,
    mode: ShotMapMode = "hybrid",
    *,
    density_max: int | None = None,
) -> go.Figure:
    """Create a binned-shot, shot-frequency, or hybrid attacking-half map.

    StatsBomb ``(x, y)`` maps to ``(y - 40, x - 60)``. Shot counts are aggregated
    into 4-by-4 coordinate-unit bins over the attacking half before being added
    to any trace. The axes focus the visible window on StatsBomb x=95..120 and
    y=10..70. Heat intensity represents shot frequency, not scoring probability.
    ``density_max`` can set a common color-scale maximum when comparing leagues.
    """
    if mode not in {"binned", "heatmap", "hybrid"}:
        raise ValueError(f"Unknown shot map mode: {mode!r}")

    prepared = prepare_shot_coordinates(shots)
    in_half = prepared.loc[prepared["in_attacking_half"]].copy()
    fig = go.Figure()

    if mode in {"heatmap", "hybrid"}:
        _add_density(fig, in_half, density_max)

    fig.update_layout(
        shapes=pitch_shapes(),
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#14251c",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 24, "r": 30, "t": 28, "b": 24},
        height=360,
        autosize=True,
        hoverlabel={
            "bgcolor": "#111a17",
            "bordercolor": "#52695a",
            "font": {"color": "#f0f3ed", "size": 12},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(17,26,23,0.82)",
            "bordercolor": "#26352e",
            "borderwidth": 1,
            "font": {"color": "#dce6dd", "size": 12},
        },
        xaxis={
            "range": list(SHOT_MAP_X_RANGE),
            "visible": False,
            "fixedrange": True,
            "constrain": "domain",
            "zeroline": False,
        },
        yaxis={
            "range": list(SHOT_MAP_Y_RANGE),
            "visible": False,
            "fixedrange": True,
            "scaleanchor": "x",
            "scaleratio": 1,
            "constrain": "domain",
            "zeroline": False,
        },
        showlegend=mode != "heatmap",
        uirevision="shot-map-pitch",
    )

    for trace in _pitch_arc_traces():
        fig.add_trace(trace)
    if mode == "binned":
        _add_shot_markers(fig, in_half, goals_only=False)
    elif mode == "hybrid":
        _add_shot_markers(fig, in_half, goals_only=True)

    return fig
