import pandas as pd
import pytest

from soccer_dashboard import data


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


def test_load_shots_rejects_invalid_shot_location(statsbomb_responses, monkeypatch):
    events = pd.DataFrame(
        [
            {
                "id": "bad-location",
                "location": None,
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

    with pytest.raises(ValueError, match="invalid location"):
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
