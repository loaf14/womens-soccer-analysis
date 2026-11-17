import pandas as pd
import pytest

from soccer_dashboard import data
from soccer_dashboard.app import create_app
from soccer_dashboard.build_data import build_shots
from soccer_dashboard.storage import load_local_shots


@pytest.fixture
def dashboard_shots():
    return pd.DataFrame(
        [
            {
                "competition_name": "NWSL",
                "season_name": "2023",
                "match_id": 1,
                "location": [110.0, 40.0],
                "in_box": True,
                "team": "NWSL United",
                "shot_outcome": "Goal",
                "shot_type": "Open Play",
                "play_pattern": "Regular Play",
                "shot_body_part": "Right Foot",
            },
            {
                "competition_name": "NWSL",
                "season_name": "2023",
                "match_id": 1,
                "location": [102.0, 18.0],
                "in_box": True,
                "team": "NWSL City",
                "shot_outcome": "Saved",
                "shot_type": "Corner",
                "play_pattern": "From Corner",
                "shot_body_part": "Left Foot",
            },
            {
                "competition_name": "NWSL",
                "season_name": "2023",
                "match_id": 2,
                "location": [45.0, 40.0],
                "in_box": False,
                "team": "NWSL United",
                "shot_outcome": "Blocked",
                "shot_type": "Penalty",
                "play_pattern": "Regular Play",
                "shot_body_part": "Head",
            },
            {
                "competition_name": "FA Women's Super League",
                "season_name": "2023/2024",
                "match_id": 1,
                "location": [115.0, 40.0],
                "in_box": True,
                "team": "London WFC",
                "shot_outcome": "Goal",
                "shot_type": "Free Kick",
                "play_pattern": "From Free Kick",
                "shot_body_part": "Right Foot",
            },
            {
                "competition_name": "FA Women's Super League",
                "season_name": "2023/2024",
                "match_id": 2,
                "location": [60.0, 0.0],
                "in_box": False,
                "team": "London WFC",
                "shot_outcome": "Off T",
                "shot_type": "Open Play",
                "play_pattern": "Regular Play",
                "shot_body_part": "Head",
            },
        ]
    )


def _components_by_id(node):
    if isinstance(node, dict):
        props = node.get("props", {})
        if "id" in props:
            yield props["id"], props
        for value in node.values():
            yield from _components_by_id(value)
    elif isinstance(node, list):
        for value in node:
            yield from _components_by_id(value)


@pytest.fixture
def statsbomb_responses(monkeypatch):
    competitions = pd.DataFrame(
        [
            {
                "competition_id": 49,
                "season_id": 107,
                "competition_name": "NWSL",
                "season_name": "2023",
            }
        ]
    )
    matches = pd.DataFrame([{"match_id": 1001}])
    events = pd.DataFrame(
        [
            {
                "id": "shot-in-box",
                "location": [110.0, 40.0],
                "shot_outcome": "Goal",
                "shot_type": "Open Play",
                "shot_body_part": "Right Foot",
                "play_pattern": "Regular Play",
                "match_id": 1001,
                "team": "Home",
                "player": "Forward",
                "type": "Shot",
            },
            {
                "id": "shot-outside-box",
                "location": [90.0, 40.0],
                "shot_outcome": "Saved",
                "shot_type": "Open Play",
                "shot_body_part": "Left Foot",
                "play_pattern": "Regular Play",
                "match_id": 1001,
                "team": "Away",
                "player": "Midfielder",
                "type": "Shot",
            },
            {
                "id": "not-a-shot",
                "location": [80.0, 30.0],
                "shot_outcome": None,
                "shot_type": None,
                "shot_body_part": None,
                "play_pattern": "Regular Play",
                "match_id": 1001,
                "team": "Home",
                "player": "Defender",
                "type": "Pass",
            },
        ]
    )

    monkeypatch.setattr(data.sb, "competitions", lambda: competitions)
    monkeypatch.setattr(
        data.sb,
        "matches",
        lambda competition_id, season_id: matches,
    )
    monkeypatch.setattr(data.sb, "events", lambda match_id: events)


def test_load_shots_returns_shot_rows_with_metadata_and_box_flag(statsbomb_responses):
    shots = data.load_shots(competition_id=49, season_id=107)

    assert list(shots.columns) == list(data.OUTPUT_COLUMNS)
    assert shots["id"].tolist() == ["shot-in-box", "shot-outside-box"]
    assert shots["competition_name"].tolist() == ["NWSL", "NWSL"]
    assert shots["season_name"].tolist() == ["2023", "2023"]
    assert shots["in_box"].tolist() == [True, False]


def test_load_shots_rejects_unknown_competition_season(statsbomb_responses):
    with pytest.raises(LookupError, match="No StatsBomb competition season"):
        data.load_shots(competition_id=49, season_id=999)


@pytest.mark.parametrize(
    "location",
    [None, [121, 40], [-1, 40], [50, 81], [50, -1], [float("nan"), 40]],
)
def test_load_shots_rejects_invalid_shot_location(
    statsbomb_responses, monkeypatch, location
):
    events = pd.DataFrame(
        [
            {
                "id": "bad-location",
                "location": location,
                "shot_outcome": "Saved",
                "shot_type": "Open Play",
                "shot_body_part": "Head",
                "play_pattern": "Regular Play",
                "match_id": 1001,
                "team": "Home",
                "player": "Forward",
                "type": "Shot",
            }
        ]
    )
    monkeypatch.setattr(data.sb, "events", lambda match_id: events)

    with pytest.raises(ValueError, match="invalid StatsBomb pitch location"):
        data.load_shots(competition_id=49, season_id=107)


def test_load_shots_rejects_missing_event_columns(statsbomb_responses, monkeypatch):
    events = pd.DataFrame(
        [
            {
                "id": "shot",
                "location": [110, 40],
                "shot_outcome": "Saved",
                "shot_type": "Open Play",
                "shot_body_part": "Head",
                "play_pattern": "Regular Play",
                "match_id": 1001,
                "team": "Home",
                "type": "Shot",
            }
        ]
    )
    monkeypatch.setattr(data.sb, "events", lambda match_id: events)

    with pytest.raises(ValueError, match="missing columns"):
        data.load_shots(competition_id=49, season_id=107)


def test_load_shots_rejects_missing_match_id_column(statsbomb_responses, monkeypatch):
    monkeypatch.setattr(
        data.sb,
        "matches",
        lambda competition_id, season_id: pd.DataFrame([{"id": 1001}]),
    )

    with pytest.raises(ValueError, match="match_id"):
        data.load_shots(competition_id=49, season_id=107)


def test_load_shots_returns_empty_schema_when_season_has_no_matches(
    statsbomb_responses, monkeypatch
):
    monkeypatch.setattr(
        data.sb,
        "matches",
        lambda competition_id, season_id: pd.DataFrame(columns=["match_id"]),
    )

    shots = data.load_shots(competition_id=49, season_id=107)

    assert shots.empty
    assert list(shots.columns) == list(data.OUTPUT_COLUMNS)


def test_build_shots_combines_seasons_and_round_trips_parquet(tmp_path, monkeypatch):
    nwsl = pd.DataFrame(
        [{column: value for column, value in zip(data.OUTPUT_COLUMNS, [
            "nwsl-shot", [110.0, 40.0], "Goal", "Open Play", "Right Foot",
            "Regular Play", 1001, "NWSL", "Forward", "NWSL", "2023", True,
        ])}]
    )
    wsl = nwsl.copy()
    wsl.loc[0, "id"] = "wsl-shot"
    wsl.loc[0, "competition_name"] = "FA Women's Super League"
    wsl.loc[0, "season_name"] = "2023/2024"
    season_data = iter([nwsl, wsl])
    requested_seasons = []

    def fake_load_shots(competition_id, season_id):
        requested_seasons.append((competition_id, season_id))
        return next(season_data)

    monkeypatch.setattr("soccer_dashboard.build_data.load_shots", fake_load_shots)
    output_path = tmp_path / "data" / "shots.parquet"

    combined = build_shots(output_path)

    assert requested_seasons == [(49, 107), (37, 281)]
    assert combined["id"].tolist() == ["nwsl-shot", "wsl-shot"]
    loaded = load_local_shots(output_path)
    assert loaded.drop(columns="location").equals(combined.drop(columns="location"))
    assert loaded["location"].map(list).tolist() == combined["location"].tolist()


def test_build_shots_can_verify_the_recorded_historical_total(tmp_path, monkeypatch):
    row_counts = iter([3530, 3529])
    monkeypatch.setattr(
        "soccer_dashboard.build_data.load_shots",
        lambda competition_id, season_id: pd.DataFrame(
            {"id": range(next(row_counts))}
        ),
    )

    shots = build_shots(tmp_path / "shots.parquet", expected_total=7059)

    assert len(shots) == 7059


def test_load_local_shots_explains_how_to_generate_missing_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="python -m soccer_dashboard.build_data"):
        load_local_shots(tmp_path / "missing.parquet")


def test_dash_layout_renders_dashboard_and_data_source(dashboard_shots):
    app = create_app(dashboard_shots)

    response = app.server.test_client().get("/_dash-layout")

    assert response.status_code == 200
    assert b"The Anatomy of a Goal" in response.data
    assert b"StatsBomb Open Data" in response.data


def test_dashboard_has_independent_league_kpi_groups(dashboard_shots):
    app = create_app(dashboard_shots)
    response = app.server.test_client().get("/_dash-layout")
    components = dict(_components_by_id(response.get_json()))

    assert components["nwsl-kpi-group"]["className"] == "league-kpi-group nwsl"
    assert components["wsl-kpi-group"]["className"] == "league-kpi-group wsl"
    assert {
        "nwsl-shots",
        "nwsl-goals",
        "nwsl-conversion-rate",
        "nwsl-matches",
        "nwsl-shots-per-match",
        "wsl-shots",
        "wsl-goals",
        "wsl-conversion-rate",
        "wsl-matches",
        "wsl-shots-per-match",
    }.issubset(components)
    assert [components[f"nwsl-{metric}"]["children"] for metric in (
        "shots",
        "goals",
        "conversion-rate",
        "matches",
        "shots-per-match",
    )] == ["3", "1", "33.3%", "2", "1.5"]
    assert [components[f"wsl-{metric}"]["children"] for metric in (
        "shots",
        "goals",
        "conversion-rate",
        "matches",
        "shots-per-match",
    )] == ["2", "1", "50.0%", "2", "1.0"]
    layout = response.get_data(as_text=True)
    assert "metric-detail" not in layout
    assert all(
        detail not in layout
        for detail in (
            "Shot events",
            "Outcome = Goal",
            "Goals ÷ shots",
            "Unique match IDs",
            "Shots ÷ matches",
        )
    )


def test_shot_map_coverage_notes_report_exclusions_per_league(dashboard_shots):
    app = create_app(dashboard_shots)
    components = dict(
        _components_by_id(app.server.test_client().get("/_dash-layout").get_json())
    )

    nwsl_note = components["nwsl-map-coverage-note"]["children"]
    wsl_note = components["wsl-map-coverage-note"]["children"]
    assert "Valid shots outside the attacking half (x < 60): 1; not plotted." in (
        nwsl_note
    )
    assert "Shot locations unavailable or invalid: 0." in nwsl_note
    assert "Valid shots outside the attacking half (x < 60): 0; not plotted." in (
        wsl_note
    )
    assert "Shot locations unavailable or invalid: 0." in wsl_note


def test_shot_map_section_copy_does_not_claim_outside_half_shots_are_plotted(
    dashboard_shots,
):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_data(as_text=True)

    assert "Compare shot locations and outcomes across both leagues." in layout
    assert "every attempt across both leagues" not in layout


def test_dashboard_has_no_global_filter_controls_or_combined_kpis(dashboard_shots):
    app = create_app(dashboard_shots)
    response = app.server.test_client().get("/_dash-layout")
    components = dict(_components_by_id(response.get_json()))

    assert {
        "competition-filter",
        "team-filter",
        "shot-outcome-filter",
        "shot-type-filter",
    }.isdisjoint(components)
    assert {
        "total-shot-count",
        "total-goal-count",
        "shot-conversion-rate",
        "match-count",
    }.isdisjoint(components)
    assert "Explore the dataset" not in response.get_data(as_text=True)


def test_dashboard_has_all_visualizations(dashboard_shots):
    app = create_app(dashboard_shots)
    response = app.server.test_client().get("/_dash-layout")
    components = dict(_components_by_id(response.get_json()))

    assert "shot-characteristics-placeholder" not in components
    assert "play-pattern-chart" in components
    assert components["play-pattern-chart"]["responsive"] is True
    assert components["play-pattern-chart"]["style"]["height"].endswith("px")
    play_pattern_figure = components["play-pattern-chart"]["figure"]
    assert [trace["name"] for trace in play_pattern_figure["data"]] == [
        "NWSL",
        "WSL",
    ]
    assert play_pattern_figure["layout"]["height"] == int(
        components["play-pattern-chart"]["style"]["height"][:-2]
    )
    assert "shot-conversion-placeholder" not in components
    assert "shot-conversion-chart" in components
    assert "sankey-all-graph" in components
    assert "sankey-goals-graph" in components
    assert "nwsl-shot-map-graph" in components
    assert "wsl-shot-map-graph" in components
    assert components["nwsl-shot-map-graph"]["style"] == {"height": "500px"}
    assert components["wsl-shot-map-graph"]["style"] == {"height": "500px"}
    assert components["nwsl-shot-map-graph"]["figure"]["layout"]["height"] == 500
    assert components["wsl-shot-map-graph"]["figure"]["layout"]["height"] == 500
    assert {
        components["nwsl-shot-map-graph"]["figure"]["data"][0]["zmax"],
        components["wsl-shot-map-graph"]["figure"]["data"][0]["zmax"],
    } == {1}
    assert components["nwsl-shot-map-title"]["children"] == "NWSL · 2023"
    assert components["wsl-shot-map-title"]["children"] == (
        "FA Women's Super League · 2023/24"
    )
    assert "equal colors indicate the same shot count per bin" in (
        response.get_data(as_text=True)
    )
    assert components["sankey-mode"]["value"] == "all"
    assert {
        option["value"] for option in components["sankey-mode"]["options"]
    } == {"all", "goals"}
    assert components["sankey-observation-count"]["children"] == (
        f"{len(dashboard_shots):,} shots represented"
    )
    sankey_figure = components["sankey-all-graph"]["figure"]
    assert sankey_figure["data"][0]["type"] == "sankey"
    assert sum(sankey_figure["data"][0]["link"]["value"]) == 3 * len(
        dashboard_shots
    )
    assert components["shot-map-view"]["value"] == "hybrid"
    assert {
        option["value"] for option in components["shot-map-view"]["options"]
    } == {"individual", "heatmap", "hybrid"}
    assert components["sankey-goals-view"]["style"] == {"display": "none"}
    assert set(
        components["sankey-goals-graph"]["figure"]["data"][0]["node"]["customdata"]
    ) == {"Play pattern", "Shot location", "Body part"}
    conversion_figure = components["shot-conversion-chart"]["figure"]
    assert components["shot-conversion-chart"]["style"] == {"height": "260px"}
    assert components["shot-conversion-chart"]["responsive"] is True
    assert conversion_figure["layout"]["height"] == 260
    assert all(trace["type"] == "scatter" for trace in conversion_figure["data"])
    conversion_traces = {
        trace["name"]: trace
        for trace in conversion_figure["data"]
        if trace.get("name")
    }
    assert {"NWSL", "WSL"}.issubset(conversion_traces)
    assert {"Inside box connector", "Outside box connector"}.issubset(
        conversion_traces
    )
    assert "Shots with missing location excluded: 0" in response.get_data(
        as_text=True
    )
    layout_text = response.get_data(as_text=True)
    assert "Where Do Shots Come From?" in layout_text
    assert "Distribution of shots by play pattern" in layout_text
    assert "missing play patterns (0) are shown as a separate category" in (
        layout_text
    )


def test_sankey_mode_callback_switches_to_goals_and_back(dashboard_shots):
    app = create_app(dashboard_shots)
    callback_output = next(
        key
        for key in app.callback_map
        if "sankey-all-view.style" in key
    )
    outputs = [
        {"id": "sankey-all-view", "property": "style"},
        {"id": "sankey-goals-view", "property": "style"},
        {"id": "sankey-observation-count", "property": "children"},
        {"id": "sankey-mode-description", "property": "children"},
    ]

    def select_mode(mode):
        response = app.server.test_client().post(
            "/_dash-update-component",
            json={
                "output": callback_output,
                "outputs": outputs,
                "inputs": [{"id": "sankey-mode", "property": "value", "value": mode}],
                "state": [],
                "changedPropIds": ["sankey-mode.value"],
            },
        )
        assert response.status_code == 200
        return response.get_json()["response"]

    goals_result = select_mode("goals")
    goal_figure = app.server.test_client().get("/_dash-layout").get_json()
    goal_components = dict(_components_by_id(goal_figure))
    goal_figure = goal_components["sankey-goals-graph"]["figure"]["data"][0]
    assert goals_result["sankey-all-view"]["style"] == {"display": "none"}
    assert goals_result["sankey-goals-view"]["style"] == {"display": "block"}
    assert set(goal_figure["node"]["customdata"]) == {
        "Play pattern",
        "Shot location",
        "Body part",
    }
    assert goals_result["sankey-observation-count"]["children"] == "2 goals represented"
    assert "goals only" in goals_result["sankey-mode-description"]["children"]
    assert "not be compared as absolute counts" in goals_result[
        "sankey-mode-description"
    ]["children"]
    assert sum(goal_figure["link"]["value"]) == 2 * 2

    all_result = select_mode("all")
    all_components = dict(
        _components_by_id(app.server.test_client().get("/_dash-layout").get_json())
    )
    all_figure = all_components["sankey-all-graph"]["figure"]["data"][0]
    assert all_result["sankey-all-view"]["style"] == {"display": "block"}
    assert all_result["sankey-goals-view"]["style"] == {"display": "none"}
    assert set(all_figure["node"]["customdata"]) == {
        "Play pattern",
        "Shot location",
        "Body part",
        "Outcome",
    }
    assert all_result["sankey-observation-count"]["children"] == (
        f"{len(dashboard_shots):,} shots represented"
    )
    assert sum(all_figure["link"]["value"]) == 3 * len(dashboard_shots)


@pytest.mark.parametrize(
    ("mode", "expected_traces", "expected_marker_counts", "expected_density_count"),
    [
        (
            "individual",
            {"Penalty arc", "Corner arcs", "Goals", "Non-goals"},
            {"Goals": 1, "Non-goals": 1},
            None,
        ),
        (
            "heatmap",
            {"Shot frequency", "Penalty arc", "Corner arcs"},
            {},
            2,
        ),
        (
            "hybrid",
            {"Shot frequency", "Penalty arc", "Corner arcs", "Goals"},
            {"Goals": 1},
            2,
        ),
    ],
)
def test_shot_map_view_callback_switches_figure(
    dashboard_shots,
    mode,
    expected_traces,
    expected_marker_counts,
    expected_density_count,
):
    app = create_app(dashboard_shots)
    callback_output = next(
        key
        for key in app.callback_map
        if "nwsl-shot-map-graph.figure" in key
    )
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": callback_output,
            "outputs": [
                {"id": "nwsl-shot-map-graph", "property": "figure"},
                {"id": "wsl-shot-map-graph", "property": "figure"},
            ],
            "inputs": [
                {"id": "shot-map-view", "property": "value", "value": mode}
            ],
            "state": [],
            "changedPropIds": ["shot-map-view.value"],
        },
    )

    assert response.status_code == 200
    result = response.json["response"]
    for league_id in ("nwsl-shot-map-graph", "wsl-shot-map-graph"):
        figure = result[league_id]["figure"]
        traces = {
            trace["name"]: trace
            for trace in figure["data"]
            if trace.get("name")
        }

        assert set(traces) == expected_traces
        for name, count in expected_marker_counts.items():
            assert len(traces[name]["customdata"]) == count
        if expected_density_count is not None:
            density = traces["Shot frequency"]
            assert sum(sum(row) for row in density["z"]) == expected_density_count

    if expected_density_count is not None:
        nwsl_density = result["nwsl-shot-map-graph"]["figure"]["data"][0]
        wsl_density = result["wsl-shot-map-graph"]["figure"]["data"][0]
        assert nwsl_density["zmax"] == wsl_density["zmax"]


def test_season_chips_are_valid_direct_header_children(dashboard_shots):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_json()
    hero = layout["props"]["children"][0]
    hero_copy = hero["props"]["children"][0]
    children = hero_copy["props"]["children"]
    season_chips = children[-1]

    assert season_chips["type"] == "Div"
    assert season_chips["props"]["className"] == "season-line"
    assert sum(
        component["props"].get("className") == "season-line"
        for component in children
    ) == 1


def test_sankey_copy_describes_categorical_associations_and_denominator(
    dashboard_shots,
):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_data(as_text=True)

    assert "How shot characteristics connect" in layout
    assert "play pattern, shot location, body part, and outcome" in layout
    assert "not goal conversion rates" in layout
    assert "Play patterns and body parts below 1% are grouped as Other" in layout
    assert "Missing outcomes are shown as Unknown" in layout
    assert "Link widths are rescaled between modes" in layout
    assert "actions leading to a shot" not in layout
    assert "How attacks unfold" not in layout


def test_dashboard_stylesheet_is_served(dashboard_shots):
    app = create_app(dashboard_shots)

    response = app.server.test_client().get("/assets/dashboard.css")

    assert response.status_code == 200
    assert b"--color-page: #0b1110" in response.data
    assert b".sankey-graph" in response.data
    assert b".sankey-graph .svg-container" in response.data
    assert b".sankey-toggle" in response.data
    assert b".conversion-chart" in response.data
    assert b".conversion-card .visualization-heading" in response.data
    assert b".play-pattern-chart" in response.data
    assert b".play-pattern-method-note" in response.data
    assert b".league-kpi-grid" in response.data
    assert b".league-kpi-metrics" in response.data
