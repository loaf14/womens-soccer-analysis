import pandas as pd
import pytest

from soccer_dashboard import data
from soccer_dashboard.app import create_app
from soccer_dashboard.build_data import build_shots
from soccer_dashboard.storage import load_local_shots


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


def test_dash_layout_renders_unfiltered_total():
    app = create_app(pd.DataFrame(index=range(7059)))

    response = app.server.test_client().get("/_dash-layout")

    assert response.status_code == 200
    assert b"7,059" in response.data
    assert b"StatsBomb Open Data" in response.data
