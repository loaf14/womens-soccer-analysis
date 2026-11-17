"""League-specific summary metrics for the dashboard header."""

from __future__ import annotations

import pandas as pd

_COMPETITIONS = {
    "NWSL": "NWSL",
    "FA Women's Super League": "WSL",
}
_REQUIRED_COLUMNS = {"competition_name", "shot_outcome", "match_id"}


def prepare_league_metrics(shots: pd.DataFrame) -> pd.DataFrame:
    """Calculate shot, goal, match, conversion, and per-match metrics by league.

    Both leagues are always returned. Empty groups receive zero counts and
    zero-valued rates so the dashboard can render them without division errors.
    """
    missing_columns = _REQUIRED_COLUMNS.difference(shots.columns)
    if missing_columns:
        raise ValueError(
            "Shot data is missing KPI columns: " + ", ".join(sorted(missing_columns))
        )

    competition_names = shots["competition_name"]
    if competition_names.isna().any():
        raise ValueError("Shot data contains missing competition names")
    unknown_competitions = set(competition_names.unique()).difference(_COMPETITIONS)
    if unknown_competitions:
        raise ValueError(
            "Unsupported competition(s) for league KPI comparison: "
            + ", ".join(sorted(unknown_competitions))
        )

    metrics = []
    for competition_name, league in _COMPETITIONS.items():
        league_shots = shots.loc[competition_names.eq(competition_name)]
        shot_count = len(league_shots)
        goal_count = int(league_shots["shot_outcome"].eq("Goal").sum())
        match_count = int(league_shots["match_id"].nunique())
        metrics.append(
            {
                "competition_name": competition_name,
                "league": league,
                "shots": shot_count,
                "goals": goal_count,
                "conversion_rate": (
                    goal_count / shot_count * 100 if shot_count else 0.0
                ),
                "matches": match_count,
                "shots_per_match": (
                    shot_count / match_count if match_count else 0.0
                ),
            }
        )

    return pd.DataFrame(
        metrics,
        columns=[
            "competition_name",
            "league",
            "shots",
            "goals",
            "conversion_rate",
            "matches",
            "shots_per_match",
        ],
    )
