"""Plotly shot-map figures using StatsBomb's 120-by-80 pitch coordinates."""

from __future__ import annotations

import math
from numbers import Real
from typing import Literal

import pandas as pd
import plotly.graph_objects as go

ShotMapMode = Literal["individual", "heatmap", "hybrid"]

PITCH_LENGTH = 120
PITCH_WIDTH = 80
HALF_PITCH_START_X = 60
HALF_PITCH_LENGTH = PITCH_LENGTH - HALF_PITCH_START_X
HALF_PITCH_WIDTH = PITCH_WIDTH
DENSITY_BIN_SIZE = 4
GOAL_COLOR = "#f0ad5e"
SHOT_COLOR = "#a9b8e8"
PITCH_LINE_COLOR = "#88a18f"
HEATMAP_COLORS = [
    [0, "#10231b"],
    [0.3, "#246342"],
    [0.68, "#49bd70"],
    [1, "#c5edaf"],
]

_TOOLTIP_FIELDS = (
    ("Player", "player"),
    ("Team", "team"),
    ("Competition", "competition_name"),
    ("Outcome", "shot_outcome"),
    ("Shot type", "shot_type"),
    ("Body part", "shot_body_part"),
    ("Play pattern", "play_pattern"),
)


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

    The attacking half is StatsBomb x=60..120. Map x is the original lateral
    coordinate centered at zero; map y is the original x coordinate offset by
    60, so attacks toward x=120 point upward on the chart.
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


def _display_value(value: object) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    text = str(value).strip()
    return text if text else "Unknown"


def _customdata(shots: pd.DataFrame) -> list[list[str]]:
    return [
        [
            *[_display_value(row.get(column)) for _, column in _TOOLTIP_FIELDS],
            f"{row['statsbomb_x']:.1f}",
            f"{row['statsbomb_y']:.1f}",
        ]
        for _, row in shots.iterrows()
    ]


_SHOT_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b> · %{customdata[1]}<br>"
    "%{customdata[2]}<br>"
    "Outcome: %{customdata[3]}<br>"
    "Shot type: %{customdata[4]}<br>"
    "Body part: %{customdata[5]}<br>"
    "Play pattern: %{customdata[6]}<br>"
    "StatsBomb location: (%{customdata[7]}, %{customdata[8]})"
    "<extra></extra>"
)


def _add_density(fig: go.Figure, shots: pd.DataFrame) -> None:
    x_start = -HALF_PITCH_WIDTH / 2
    x_bin_count = int(HALF_PITCH_WIDTH / DENSITY_BIN_SIZE)
    y_bin_count = int(HALF_PITCH_LENGTH / DENSITY_BIN_SIZE)
    counts = [[0] * y_bin_count for _ in range(x_bin_count)]
    for x, y in zip(shots["map_x"], shots["map_y"], strict=True):
        # Clamp the maximum pitch edges into the final bins.
        x_bin = min(int((x - x_start) / DENSITY_BIN_SIZE), x_bin_count - 1)
        y_bin = min(int(y / DENSITY_BIN_SIZE), y_bin_count - 1)
        counts[x_bin][y_bin] += 1

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
            zmax=max(1, max(map(max, counts))),
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
    if goals_only:
        groups = [(True, "Goals", GOAL_COLOR, 11)]
    else:
        groups = [
            (False, "Non-goals", SHOT_COLOR, 7),
            (True, "Goals", GOAL_COLOR, 11),
        ]

    for is_goal, label, color, size in groups:
        group = shots.loc[shots["is_goal"].eq(is_goal)]
        fig.add_trace(
            go.Scattergl(
                x=group["map_x"],
                y=group["map_y"],
                mode="markers",
                name=label,
                marker={
                    "color": color,
                    "size": size,
                    "opacity": 0.86 if is_goal else 0.48,
                    "line": {
                        "color": "#f8f2e7" if is_goal else "rgba(0,0,0,0)",
                        "width": 1 if is_goal else 0,
                    },
                },
                customdata=_customdata(group),
                hovertemplate=_SHOT_HOVER_TEMPLATE,
            )
        )


def create_shot_map(
    shots: pd.DataFrame, mode: ShotMapMode = "hybrid"
) -> go.Figure:
    """Create an individual, shot-frequency, or hybrid attacking-half map.

    The attacking-half view maps StatsBomb ``(x, y)`` to ``(y - 40, x - 60)``.
    Valid shots with ``x < 60`` and shots with unavailable/invalid locations
    are not silently included or repositioned; the dashboard reports these
    counts in its coverage note. Density uses 4-by-4 coordinate-unit histogram
    bins and represents shot frequency, not scoring probability. The upper
    pitch boundaries are included in the final bins.
    """
    if mode not in {"individual", "heatmap", "hybrid"}:
        raise ValueError(f"Unknown shot map mode: {mode!r}")

    prepared = prepare_shot_coordinates(shots)
    in_half = prepared.loc[prepared["in_attacking_half"]].copy()
    fig = go.Figure()

    if mode in {"heatmap", "hybrid"}:
        _add_density(fig, in_half)

    fig.update_layout(
        shapes=pitch_shapes(),
        template="plotly_dark",
        paper_bgcolor="#111a17",
        plot_bgcolor="#14251c",
        font={"family": "DM Sans, Segoe UI, sans-serif", "color": "#f0f3ed"},
        margin={"l": 24, "r": 30, "t": 28, "b": 24},
        height=680,
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
            "range": [-41, 41],
            "visible": False,
            "fixedrange": True,
            "constrain": "domain",
            "zeroline": False,
        },
        yaxis={
            "range": [0, 62],
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
    if mode == "individual":
        _add_shot_markers(fig, in_half, goals_only=False)
    elif mode == "hybrid":
        _add_shot_markers(fig, in_half, goals_only=True)

    return fig
