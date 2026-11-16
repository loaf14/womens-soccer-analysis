import pandas as pd
import pytest

from soccer_dashboard.shot_map import (
    DENSITY_BIN_SIZE,
    create_shot_map,
    pitch_shapes,
    prepare_shot_coordinates,
)


@pytest.fixture
def shot_map_shots():
    return pd.DataFrame(
        [
            {
                "id": "goal-central",
                "location": [108.0, 40.0],
                "shot_outcome": "Goal",
                "player": "Forward",
                "team": "NWSL City",
                "competition_name": "NWSL",
                "shot_type": "Open Play",
                "shot_body_part": "Right Foot",
                "play_pattern": "Regular Play",
            },
            {
                "id": "save-left",
                "location": [102.0, 18.0],
                "shot_outcome": "Saved",
                "player": None,
                "team": "London WFC",
                "competition_name": "FA Women's Super League",
                "shot_type": "Free Kick",
                "shot_body_part": "Left Foot",
                "play_pattern": "From Free Kick",
            },
            {
                "id": "goal-edge",
                "location": [60.0, 0.0],
                "shot_outcome": "Goal",
                "player": "Winger",
                "team": "NWSL United",
                "competition_name": "NWSL",
                "shot_type": "Penalty",
                "shot_body_part": "Right Foot",
                "play_pattern": "From Penalty",
            },
            {
                "id": "outside-half",
                "location": [45.0, 40.0],
                "shot_outcome": "Blocked",
                "player": "Midfielder",
                "team": "NWSL United",
                "competition_name": "NWSL",
                "shot_type": "Open Play",
                "shot_body_part": "Left Foot",
                "play_pattern": "Regular Play",
            },
            {
                "id": "missing-location",
                "location": None,
                "shot_outcome": "Off T",
                "player": "Striker",
                "team": "London WFC",
                "competition_name": "FA Women's Super League",
                "shot_type": "Open Play",
                "shot_body_part": "Head",
                "play_pattern": "Regular Play",
            },
        ]
    )


def _trace_by_name(figure, name):
    return next(trace for trace in figure.data if trace.name == name)


def test_coordinate_transform_preserves_orientation_and_lateral_axis(shot_map_shots):
    prepared = prepare_shot_coordinates(shot_map_shots)

    central = prepared.loc[prepared["id"].eq("goal-central")].iloc[0]
    left = prepared.loc[prepared["id"].eq("save-left")].iloc[0]
    edge = prepared.loc[prepared["id"].eq("goal-edge")].iloc[0]

    assert (central["statsbomb_x"], central["statsbomb_y"]) == (108.0, 40.0)
    assert (central["map_x"], central["map_y"]) == (0.0, 48.0)
    assert (left["map_x"], left["map_y"]) == (-22.0, 42.0)
    assert (edge["map_x"], edge["map_y"]) == (-40.0, 0.0)
    assert central["map_y"] < 60


def test_pitch_geometry_matches_statsbomb_half_pitch_dimensions():
    shapes = {shape["name"]: shape for shape in pitch_shapes()}

    assert (
        shapes["pitch-boundary"]["x0"],
        shapes["pitch-boundary"]["x1"],
        shapes["pitch-boundary"]["y0"],
        shapes["pitch-boundary"]["y1"],
    ) == (-40, 40, 0, 60)
    assert (
        shapes["penalty-area"]["x0"],
        shapes["penalty-area"]["x1"],
        shapes["penalty-area"]["y0"],
        shapes["penalty-area"]["y1"],
    ) == (-22, 22, 42, 60)
    assert (
        shapes["six-yard-box"]["x0"],
        shapes["six-yard-box"]["x1"],
        shapes["six-yard-box"]["y0"],
        shapes["six-yard-box"]["y1"],
    ) == (-10, 10, 54, 60)
    assert (
        shapes["goal"]["x0"],
        shapes["goal"]["x1"],
        shapes["goal"]["y0"],
        shapes["goal"]["y1"],
    ) == (-4, 4, 60, 62)
    assert shapes["center-circle"]["x1"] == pytest.approx(9.15)
    assert shapes["penalty-spot"]["y0"] == pytest.approx(47.65)

    figure = create_shot_map(pd.DataFrame({"location": []}))
    assert figure.layout.yaxis.scaleanchor == "x"
    assert figure.layout.yaxis.range[1] > figure.layout.yaxis.range[0]
    assert "Penalty arc" in {trace.name for trace in figure.data}
    assert "Corner arcs" in {trace.name for trace in figure.data}


def test_individual_mode_separates_goals_and_non_goals_and_counts_mapped_shots(
    shot_map_shots,
):
    figure = create_shot_map(shot_map_shots, mode="individual")
    goals = _trace_by_name(figure, "Goals")
    non_goals = _trace_by_name(figure, "Non-goals")

    assert len(goals.x) == 2
    assert len(non_goals.x) == 1
    assert len(goals.x) + len(non_goals.x) == 3
    assert list(goals.x) == [0.0, -40.0]
    assert goals.marker.color == "#f0ad5e"
    assert non_goals.marker.color == "#a9b8e8"


def test_heatmap_uses_every_valid_shot_in_attacking_half_and_documented_bins(
    shot_map_shots,
):
    figure = create_shot_map(shot_map_shots, mode="heatmap")
    density = _trace_by_name(figure, "Shot frequency")

    assert density.type == "histogram2d"
    assert len(density.x) == len(density.y) == 3
    assert density.xbins.size == DENSITY_BIN_SIZE
    assert density.ybins.size == DENSITY_BIN_SIZE
    assert list(density.x) == [0.0, -22.0, -40.0]
    assert list(density.y) == [48.0, 42.0, 0.0]
    assert density.colorbar.title.text == "Shot attempts per bin"
    assert len(density.colorscale) > 1
    assert int(shot_map_shots["shot_outcome"].eq("Goal").sum()) == 2
    assert len(density.x) == 3


def test_hybrid_uses_same_all_shot_density_and_adds_only_goals(shot_map_shots):
    heatmap = create_shot_map(shot_map_shots, mode="heatmap")
    hybrid = create_shot_map(shot_map_shots, mode="hybrid")
    heatmap_density = _trace_by_name(heatmap, "Shot frequency")
    hybrid_density = _trace_by_name(hybrid, "Shot frequency")
    goals = _trace_by_name(hybrid, "Goals")

    assert list(hybrid_density.x) == list(heatmap_density.x)
    assert list(hybrid_density.y) == list(heatmap_density.y)
    assert len(goals.x) == 2
    assert "Non-goals" not in {trace.name for trace in hybrid.data}
    assert list(hybrid.data).index(hybrid_density) < list(hybrid.data).index(goals)


def test_hybrid_is_default_mode(shot_map_shots):
    figure = create_shot_map(shot_map_shots)

    assert "Shot frequency" in {trace.name for trace in figure.data}
    assert "Goals" in {trace.name for trace in figure.data}
    assert "Non-goals" not in {trace.name for trace in figure.data}


def test_invalid_optional_tooltip_values_are_rendered_as_unknown(shot_map_shots):
    figure = create_shot_map(shot_map_shots, mode="individual")
    non_goals = _trace_by_name(figure, "Non-goals")

    assert non_goals.customdata[0][0] == "Unknown"
    assert non_goals.hovertemplate


@pytest.mark.parametrize("mode", ["individual", "heatmap", "hybrid"])
def test_empty_shots_produce_valid_empty_figures(mode):
    empty_shots = pd.DataFrame(
        columns=[
            "location",
            "shot_outcome",
            "player",
            "team",
            "competition_name",
            "shot_type",
        ]
    )

    figure = create_shot_map(empty_shots, mode=mode)

    assert figure.layout.shapes
    if mode in {"heatmap", "hybrid"}:
        assert len(_trace_by_name(figure, "Shot frequency").x) == 0
    if mode in {"individual", "hybrid"}:
        assert len(_trace_by_name(figure, "Goals").x) == 0


def test_outside_half_and_invalid_locations_are_counted_not_repositioned(
    shot_map_shots,
):
    prepared = prepare_shot_coordinates(shot_map_shots)

    outside = prepared["has_valid_location"] & ~prepared["in_attacking_half"]
    unavailable = ~prepared["has_valid_location"]
    mapped = prepared.loc[prepared["in_attacking_half"]]

    assert int(outside.sum()) == 1
    assert int(unavailable.sum()) == 1
    assert len(mapped) == 3
    assert "outside-half" not in mapped["id"].tolist()
    assert "missing-location" not in mapped["id"].tolist()
