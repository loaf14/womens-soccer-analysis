import pandas as pd
import pytest

from soccer_dashboard.shot_map import (
    DENSITY_BIN_SIZE,
    SHOT_MAP_X_RANGE,
    SHOT_MAP_Y_RANGE,
    create_shot_map,
    max_shot_density,
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
    assert tuple(figure.layout.xaxis.range) == SHOT_MAP_X_RANGE
    assert tuple(figure.layout.yaxis.range) == SHOT_MAP_Y_RANGE
    assert figure.layout.yaxis.range[1] > figure.layout.yaxis.range[0]
    figure_shapes = {shape.name: shape for shape in figure.layout.shapes}
    for shape_name in ("goal", "penalty-area", "six-yard-box"):
        assert figure_shapes[shape_name].y0 >= figure.layout.yaxis.range[0]
        assert figure_shapes[shape_name].y1 <= figure.layout.yaxis.range[1]
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

    assert density.type == "heatmap"
    assert len(density.x) == 2 * 40 / DENSITY_BIN_SIZE
    assert len(density.y) == 60 / DENSITY_BIN_SIZE
    assert len(density.z) == len(density.y)
    assert len(density.z[0]) == len(density.x)
    assert density.colorbar.title.text == "Shot attempts per bin"
    assert len(density.colorscale) > 1
    assert sum(map(sum, density.z)) == 3


def test_density_max_can_be_shared_without_independent_normalization():
    nwsl = pd.DataFrame(
        {
            "location": [[108.0, 40.0]] * 3 + [[45.0, 40.0], None],
            "shot_outcome": ["Saved"] * 5,
        }
    )
    wsl = pd.DataFrame(
        {
            "location": [[110.0, 42.0]] * 2,
            "shot_outcome": ["Goal", "Saved"],
        }
    )
    shared_max = max(max_shot_density(nwsl), max_shot_density(wsl))

    nwsl_figure = create_shot_map(nwsl, mode="heatmap", density_max=shared_max)
    wsl_figure = create_shot_map(wsl, mode="heatmap", density_max=shared_max)
    nwsl_density = _trace_by_name(nwsl_figure, "Shot frequency")
    wsl_density = _trace_by_name(wsl_figure, "Shot frequency")

    assert max_shot_density(nwsl) == 3
    assert max_shot_density(wsl) == 2
    assert nwsl_density.zmax == wsl_density.zmax == 3
    assert list(nwsl_density.x) == list(wsl_density.x)
    assert list(nwsl_density.y) == list(wsl_density.y)
    assert nwsl_density.colorscale == wsl_density.colorscale
    assert sum(map(sum, nwsl_density.z)) == 3
    assert sum(map(sum, wsl_density.z)) == 2


def test_shared_density_max_rejects_a_scale_below_local_peak(shot_map_shots):
    with pytest.raises(ValueError, match="cannot be lower"):
        create_shot_map(shot_map_shots, mode="heatmap", density_max=0)


@pytest.mark.parametrize("mode", ["heatmap", "hybrid"])
def test_density_bin_counts_conserve_attacking_half_shots_at_all_boundaries(mode):
    shots = pd.DataFrame(
        {
            "location": [
                [60.0, 0.0],
                [60.0, 80.0],
                [120.0, 0.0],
                [120.0, 80.0],
                [108.0, 40.0],
                [45.0, 40.0],
                None,
            ],
            "shot_outcome": ["Saved", "Saved", "Saved", "Goal", "Blocked", "Saved", "Saved"],
        }
    )
    expected_count = int(prepare_shot_coordinates(shots)["in_attacking_half"].sum())
    density = _trace_by_name(create_shot_map(shots, mode=mode), "Shot frequency")

    assert sum(map(sum, density.z)) == expected_count == 5
    assert density.z[0][0] == 1
    assert density.z[0][-1] == 1
    assert density.z[-1][0] == 1
    assert density.z[-1][-1] == 1


def test_hybrid_uses_same_all_shot_density_and_adds_only_goals(shot_map_shots):
    heatmap = create_shot_map(shot_map_shots, mode="heatmap")
    hybrid = create_shot_map(shot_map_shots, mode="hybrid")
    heatmap_density = _trace_by_name(heatmap, "Shot frequency")
    hybrid_density = _trace_by_name(hybrid, "Shot frequency")
    goals = _trace_by_name(hybrid, "Goals")

    assert [list(row) for row in hybrid_density.z] == [
        list(row) for row in heatmap_density.z
    ]
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
        assert sum(map(sum, _trace_by_name(figure, "Shot frequency").z)) == 0
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


def test_zoomed_view_flags_outside_window_without_dropping_shot_rows(shot_map_shots):
    prepared = prepare_shot_coordinates(shot_map_shots)

    assert len(prepared) == len(shot_map_shots)
    assert prepared["in_shot_map_view"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
