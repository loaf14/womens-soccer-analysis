import pandas as pd
import pytest

from soccer_dashboard import data
from soccer_dashboard.app import create_app, create_server
from soccer_dashboard.build_data import build_shots
from soccer_dashboard.storage import load_local_shots


@pytest.fixture
def dashboard_shots():
    shots = pd.DataFrame(
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
    shots["shot_key_pass_id"] = ["pass-1", None, None, "pass-2", None]
    shots["assisted"] = shots["shot_key_pass_id"].notna()
    return shots


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


def _nodes_by_class(node, class_name):
    if isinstance(node, dict):
        if node.get("props", {}).get("className") == class_name:
            yield node
        for value in node.values():
            yield from _nodes_by_class(value, class_name)
    elif isinstance(node, list):
        for value in node:
            yield from _nodes_by_class(value, class_name)


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
                "shot_key_pass_id": "pass-1",
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
                "shot_key_pass_id": None,
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
                "shot_key_pass_id": None,
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
    assert shots["shot_key_pass_id"].iloc[0] == "pass-1"
    assert pd.isna(shots["shot_key_pass_id"].iloc[1])
    assert shots["assisted"].tolist() == [True, False]


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
                "shot_key_pass_id": None,
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
            "Regular Play", "pass-1", 1001, "NWSL", "Forward", "NWSL",
            "2023", True, True,
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
    assert loaded["assisted"].tolist() == [True, True]


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
    layout = response.get_data(as_text=True)
    assert "SHOT DISTRIBUTION" in layout
    assert "Where teams take their shots" in layout
    assert (
        "Compare the spatial distribution and concentration of shot attempts "
        "across leagues."
    ) in layout
    assert b"shot-map-view" not in response.data
    assert b"Hybrid" not in response.data
    assert b"Binned shots" not in response.data
    assert b"Individual shots" not in response.data


def test_gunicorn_factory_uses_prepared_data_without_api_requests(
    dashboard_shots, monkeypatch
):
    load_calls = []

    def load_prepared_shots():
        load_calls.append(True)
        return dashboard_shots

    monkeypatch.setattr(
        "soccer_dashboard.app.load_local_shots", load_prepared_shots
    )
    for method in ("competitions", "matches", "events"):
        monkeypatch.setattr(
            data.sb,
            method,
            lambda *args, **kwargs: pytest.fail(
                f"Unexpected StatsBomb API request through {method}"
            ),
        )

    server = create_server()
    client = server.test_client()

    assert load_calls == [True]
    assert client.get("/_dash-layout").status_code == 200
    dataset_url = client.get("/data/shots.parquet")
    assert dataset_url.mimetype == "text/html"
    assert not dataset_url.data.startswith(b"PAR1")
    logo = client.get("/assets/statsbomb-logo.svg")
    assert logo.status_code == 200
    assert b"<svg" in logo.data


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


def test_shot_map_section_copy_does_not_claim_outside_half_shots_are_plotted(
    dashboard_shots,
):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_data(as_text=True)

    assert "SHOT DISTRIBUTION" in layout
    assert "Where teams take their shots" in layout
    assert (
        "Compare the spatial distribution and concentration of shot attempts "
        "across leagues."
    ) in layout
    assert "Compare shot locations and outcomes across both leagues." not in layout
    assert "shot-map-view" not in layout
    assert "Hybrid" not in layout
    assert "Binned shots" not in layout
    assert "Individual shots" not in layout
    assert "every attempt across both leagues" not in layout
    assert '"children":"Compare shot locations"' not in layout
    assert "zoomed penalty-area view" not in layout
    assert "Heat intensity represents shot frequency" not in layout
    assert "Focused view: StatsBomb" not in layout
    assert "Valid shots outside this window" not in layout


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
    assert "play-pattern-chart" not in components
    assert "shot-conversion-chart" not in components
    assert "play-pattern-conversion-chart" in components
    assert "assist-conversion-chart" in components
    assert components["play-pattern-conversion-chart"]["className"] == (
        "conversion-comparison-chart"
    )
    assert components["assist-conversion-chart"]["className"] == (
        "conversion-comparison-chart"
    )
    assert "nwsl-sankey-graph" in components
    assert "wsl-sankey-graph" in components
    assert "sankey-all-graph" not in components
    assert "sankey-goals-graph" not in components
    assert "nwsl-shot-map-graph" in components
    assert "wsl-shot-map-graph" in components
    assert "shot-map-view" not in components
    assert not any(
        "shot-map-graph.figure" in callback
        for callback in app.callback_map
    )
    assert components["nwsl-shot-map-graph"]["style"] == {"height": "360px"}
    assert components["wsl-shot-map-graph"]["style"] == {"height": "360px"}
    assert components["nwsl-shot-map-graph"]["figure"]["layout"]["height"] == 360
    assert components["wsl-shot-map-graph"]["figure"]["layout"]["height"] == 360
    for graph_id in ("nwsl-shot-map-graph", "wsl-shot-map-graph"):
        figure_traces = components[graph_id]["figure"]["data"]
        assert [trace["type"] for trace in figure_traces] == [
            "heatmap",
            "scatter",
            "scatter",
        ]
        assert not any(trace.get("mode") == "markers" for trace in figure_traces)
    assert {
        components["nwsl-shot-map-graph"]["figure"]["data"][0]["zmax"],
        components["wsl-shot-map-graph"]["figure"]["data"][0]["zmax"],
    } == {1}
    assert components["nwsl-shot-map-title"]["children"] == "NWSL · 2023"
    assert components["wsl-shot-map-title"]["children"] == "WSL · 2023/24"
    assert components["sankey-mode"]["value"] == "all"
    assert {
        option["value"] for option in components["sankey-mode"]["options"]
    } == {"all", "goals"}
    assert components["nwsl-sankey-count"]["children"] == "3 shots represented"
    assert components["wsl-sankey-count"]["children"] == "2 shots represented"
    assert components["nwsl-sankey-title"]["children"] == "NWSL · 2023"
    assert components["wsl-sankey-title"]["children"] == "WSL · 2023/24"
    assert components["nwsl-sankey-graph"]["style"] == {"height": "500px"}
    assert components["wsl-sankey-graph"]["style"] == {"height": "500px"}
    nwsl_sankey = components["nwsl-sankey-graph"]["figure"]["data"][0]
    wsl_sankey = components["wsl-sankey-graph"]["figure"]["data"][0]
    for trace, shot_count in ((nwsl_sankey, 3), (wsl_sankey, 2)):
        assert trace["type"] == "sankey"
        assert sum(trace["link"]["value"]) == 3 * shot_count
        assert "Source category" in trace["link"]["hovertemplate"]
        assert "Source category total" in trace["link"]["hovertemplate"]
        assert list(dict.fromkeys(trace["node"]["customdata"])) == [
            "Shot location",
            "Play pattern",
            "Body part",
            "Outcome",
        ]
    assert "From Corner" in nwsl_sankey["node"]["label"]
    assert "From Corner" not in wsl_sankey["node"]["label"]
    assert "From Free Kick" not in nwsl_sankey["node"]["label"]
    assert "From Free Kick" in wsl_sankey["node"]["label"]
    assert components["sankey-comparison-grid"]["className"] == (
        "sankey-comparison-grid"
    )
    assert components["nwsl-sankey-graph"]["config"]["responsive"] is True
    assert components["wsl-sankey-graph"]["config"]["responsive"] is True
    pattern_figure = components["play-pattern-conversion-chart"]["figure"]
    assist_figure = components["assist-conversion-chart"]["figure"]
    for chart_id, figure in (
        ("play-pattern-conversion-chart", pattern_figure),
        ("assist-conversion-chart", assist_figure),
    ):
        assert components[chart_id]["responsive"] is True
        assert int(components[chart_id]["style"]["height"][:-2]) == (
            figure["layout"]["height"]
        )
    assert {trace["name"] for trace in pattern_figure["data"]} >= {
        "NWSL",
        "WSL",
    }
    assert [trace["type"] for trace in assist_figure["data"]] == ["bar", "bar"]
    assert [trace["name"] for trace in assist_figure["data"]] == ["NWSL", "WSL"]
    pattern_traces = {
        trace["name"]: trace
        for trace in pattern_figure["data"]
        if trace.get("name") in {"NWSL", "WSL"}
    }
    assert pattern_traces["NWSL"]["marker"]["color"] == "#e7a45e"
    assert pattern_traces["WSL"]["marker"]["color"] == "#8db4ff"
    assert assist_figure["data"][0]["marker"]["color"] == "#e7a45e"
    assert assist_figure["data"][1]["marker"]["color"] == "#8db4ff"
    layout_text = response.get_data(as_text=True)
    assert "GOAL-SCORING EFFICIENCY" in layout_text
    assert "What influences shot conversion?" in layout_text
    assert "Compare goal conversion across play patterns" in layout_text
    assert "Conversion by play pattern" in layout_text
    assert "Conversion by assist status" in layout_text
    assert "no recorded key pass" in layout_text
    assert "Other play patterns excluded from this comparison." in layout_text
    assert "Other" not in set(
        pattern_traces["NWSL"]["y"] + pattern_traces["WSL"]["y"]
    )


def test_sankey_mode_callback_switches_to_goals_and_back(dashboard_shots):
    app = create_app(dashboard_shots)
    callback_output = next(
        key
        for key in app.callback_map
        if "nwsl-sankey-graph.figure" in key
    )
    outputs = [
        {"id": "nwsl-sankey-graph", "property": "figure"},
        {"id": "wsl-sankey-graph", "property": "figure"},
        {"id": "nwsl-sankey-count", "property": "children"},
        {"id": "wsl-sankey-count", "property": "children"},
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
    for graph_id, shot_count in (
        ("nwsl-sankey-graph", 1),
        ("wsl-sankey-graph", 1),
    ):
        goal_trace = goals_result[graph_id]["figure"]["data"][0]
        assert list(dict.fromkeys(goal_trace["node"]["customdata"])) == [
            "Shot location",
            "Play pattern",
            "Body part",
        ]
        assert sum(goal_trace["link"]["value"]) == 2 * shot_count
    assert goals_result["nwsl-sankey-count"]["children"] == "1 goals represented"
    assert goals_result["wsl-sankey-count"]["children"] == "1 goals represented"

    all_result = select_mode("all")
    for graph_id, shot_count in (
        ("nwsl-sankey-graph", 3),
        ("wsl-sankey-graph", 2),
    ):
        all_trace = all_result[graph_id]["figure"]["data"][0]
        assert list(dict.fromkeys(all_trace["node"]["customdata"])) == [
            "Shot location",
            "Play pattern",
            "Body part",
            "Outcome",
        ]
        assert sum(all_trace["link"]["value"]) == 3 * shot_count
    assert all_result["nwsl-sankey-count"]["children"] == "3 shots represented"
    assert all_result["wsl-sankey-count"]["children"] == "2 shots represented"


def test_season_chips_are_valid_direct_header_children(dashboard_shots):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_json()
    hero = layout["props"]["children"][0]
    hero_copy = hero["props"]["children"][0]
    children = hero_copy["props"]["children"]
    season_chips = children[-1]

    assert season_chips["type"] == "Div"
    assert season_chips["props"]["className"] == "season-line"
    assert [child["props"]["children"] for child in season_chips["props"]["children"]] == [
        "NWSL · 2023",
        "FA Women's Super League · 2023/2024",
    ]
    assert [
        child["props"]["className"] for child in season_chips["props"]["children"]
    ] == ["season-chip nwsl", "season-chip wsl"]
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
    assert "Compare shot patterns and outcomes across leagues." in layout
    assert "Link widths show shot counts" in layout
    assert "each league diagram is scaled independently" in layout
    assert "share of the source category, not goal conversion rates" in layout
    assert "Play patterns and body parts below 1% are grouped as Other" not in layout
    assert "Missing outcomes are shown as Unknown" not in layout
    assert "actions leading to a shot" not in layout
    assert "How attacks unfold" not in layout


def test_sankey_toolbar_places_note_and_controls_in_one_row(dashboard_shots):
    app = create_app(dashboard_shots)
    layout = app.server.test_client().get("/_dash-layout").get_json()

    toolbar = next(_nodes_by_class(layout, "sankey-toolbar"))
    note, controls = toolbar["props"]["children"]

    assert note["props"]["className"] == "sankey-method-note"
    assert controls["props"]["className"] == "sankey-toolbar-controls"
    assert [child["type"] for child in controls["props"]["children"]] == [
        "Label",
        "RadioItems",
    ]


def test_dashboard_stylesheet_is_served(dashboard_shots):
    app = create_app(dashboard_shots)

    response = app.server.test_client().get("/assets/dashboard.css")

    assert response.status_code == 200
    stylesheet = response.get_data(as_text=True).replace("\r\n", "\n")
    assert b"--color-page: #0b1110" in response.data
    assert b".sankey-graph" in response.data
    assert b".sankey-graph .svg-container" in response.data
    assert (
        ".sankey-toolbar {\n"
        "  display: flex;\n"
        "  align-items: center;\n"
        "  justify-content: space-between;\n"
        "  gap: 1.5rem;\n"
        "  margin-bottom: 0.75rem;\n"
        "  padding: 0;\n"
        "}"
    ) in stylesheet
    assert ".sankey-toggle .shot-map-option {\n  gap: 0.5rem;\n}" in stylesheet
    assert b".sankey-toggle" in response.data
    assert (
        ".shot-map-radio {\n"
        "  width: 0.85rem;\n"
        "  height: 0.85rem;\n"
        "  margin: 0 8px 0 0;\n"
        "  accent-color: var(--color-pitch);\n"
        "  vertical-align: middle;\n"
        "}"
    ) in stylesheet
    assert "@media (max-width: 900px)" in stylesheet
    assert (
        ".sankey-toolbar {\n"
        "    align-items: stretch;\n"
        "    flex-direction: column;\n"
        "    gap: 0.75rem;\n"
        "  }"
    ) in stylesheet
    assert b".sankey-comparison-grid" in response.data
    assert b"@media (max-width: 1100px)" in response.data
    assert b"grid-template-columns: minmax(0, 1fr)" in response.data
    assert b".sankey-league-panel" in response.data
    assert b".conversion-comparison-chart" in response.data
    assert b".conversion-chart-note" in response.data
    assert b".goal-conversion-grid" in response.data
    assert (
        ".goal-conversion-grid {\n"
        "    grid-template-columns: minmax(0, 1fr);\n"
        "  }"
    ) in stylesheet
    assert b".league-kpi-grid" in response.data
    assert b".league-kpi-metrics" in response.data
