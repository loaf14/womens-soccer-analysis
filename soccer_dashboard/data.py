"""StatsBomb Open Data integration and shot-level data preparation."""

from __future__ import annotations

import math
from numbers import Real

import pandas as pd
from statsbombpy import sb

SHOT_COLUMNS = (
    "id",
    "location",
    "shot_outcome",
    "shot_type",
    "shot_body_part",
    "play_pattern",
    "match_id",
    "team",
    "player",
)
OUTPUT_COLUMNS = (*SHOT_COLUMNS, "competition_name", "season_name", "in_box")
EVENT_COLUMNS = ("type", *SHOT_COLUMNS)


def _location_is_in_box(location: object, event_id: object) -> bool:
    if (
        not isinstance(location, (list, tuple))
        or len(location) < 2
        or not isinstance(location[0], Real)
        or not isinstance(location[1], Real)
        or not math.isfinite(float(location[0]))
        or not math.isfinite(float(location[1]))
        or not 0 <= location[0] <= 120
        or not 0 <= location[1] <= 80
    ):
        raise ValueError(
            f"Shot event {event_id!r} has an invalid StatsBomb pitch location: "
            f"{location!r}"
        )

    x, y = location[:2]
    return x >= 102 and 18 <= y <= 62


def load_shots(competition_id: int, season_id: int) -> pd.DataFrame:
    """Fetch all shot events for a StatsBomb competition season.

    The returned ``in_box`` value identifies shots at x >= 102 and 18 <= y <= 62
    on StatsBomb's 120-by-80 pitch. Missing, malformed, or out-of-bounds shot
    coordinates raise ``ValueError`` instead of being silently counted as
    outside the box.
    """
    competitions = sb.competitions()
    required_metadata = {"competition_id", "season_id", "competition_name", "season_name"}
    missing_metadata = required_metadata.difference(competitions.columns)
    if missing_metadata:
        raise ValueError(
            "StatsBomb competitions response is missing columns: "
            f"{sorted(missing_metadata)}"
        )

    selected = competitions.loc[
        competitions["competition_id"].eq(competition_id)
        & competitions["season_id"].eq(season_id)
    ]
    if selected.empty:
        raise LookupError(
            f"No StatsBomb competition season found for "
            f"competition_id={competition_id}, season_id={season_id}"
        )
    if len(selected) != 1:
        raise ValueError(
            f"Expected one competition season for competition_id={competition_id}, "
            f"season_id={season_id}; found {len(selected)}"
        )

    competition_name = selected.iloc[0]["competition_name"]
    season_name = selected.iloc[0]["season_name"]
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    if "match_id" not in matches.columns:
        raise ValueError("StatsBomb matches response is missing the 'match_id' column")

    shot_frames = []
    for match_id in matches["match_id"]:
        if pd.isna(match_id):
            raise ValueError("StatsBomb matches response contains a missing match_id")

        events = sb.events(match_id=match_id)
        missing_event_columns = set(EVENT_COLUMNS).difference(events.columns)
        if missing_event_columns:
            raise ValueError(
                f"StatsBomb events response for match {match_id!r} is missing columns: "
                f"{sorted(missing_event_columns)}"
            )

        shots = events.loc[events["type"].eq("Shot"), list(SHOT_COLUMNS)].copy()
        shots["competition_name"] = competition_name
        shots["season_name"] = season_name
        shots["in_box"] = [
            _location_is_in_box(location, event_id)
            for location, event_id in zip(shots["location"], shots["id"])
        ]
        shot_frames.append(shots)

    if not shot_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.concat(shot_frames, ignore_index=True).loc[:, OUTPUT_COLUMNS]
