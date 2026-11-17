import pandas as pd
import pytest

from soccer_dashboard.league_kpis import prepare_league_metrics


@pytest.fixture
def league_shots():
    return pd.DataFrame(
        [
            {"competition_name": "NWSL", "match_id": 1, "shot_outcome": "Goal"},
            {"competition_name": "NWSL", "match_id": 1, "shot_outcome": "Saved"},
            {"competition_name": "NWSL", "match_id": 2, "shot_outcome": "Goal"},
            {
                "competition_name": "FA Women's Super League",
                "match_id": 1,
                "shot_outcome": "Goal",
            },
            {
                "competition_name": "FA Women's Super League",
                "match_id": 2,
                "shot_outcome": "Saved",
            },
        ]
    )


def _league(metrics, name):
    return metrics.loc[metrics["league"].eq(name)].iloc[0]


def test_league_metrics_calculate_counts_and_rates_independently(league_shots):
    metrics = prepare_league_metrics(league_shots)
    nwsl = _league(metrics, "NWSL")
    wsl = _league(metrics, "WSL")

    assert tuple(nwsl[["shots", "goals", "matches"]]) == (3, 2, 2)
    assert nwsl["conversion_rate"] == pytest.approx(200 / 3)
    assert nwsl["shots_per_match"] == pytest.approx(1.5)
    assert tuple(wsl[["shots", "goals", "matches"]]) == (2, 1, 2)
    assert wsl["conversion_rate"] == pytest.approx(50)
    assert wsl["shots_per_match"] == pytest.approx(1)


def test_empty_league_is_returned_with_safe_zero_rates(league_shots):
    nwsl_only = league_shots.loc[league_shots["competition_name"].eq("NWSL")]
    metrics = prepare_league_metrics(nwsl_only)
    wsl = _league(metrics, "WSL")

    assert (wsl[["shots", "goals", "matches"]].astype(int) == 0).all()
    assert wsl["conversion_rate"] == 0
    assert wsl["shots_per_match"] == 0


def test_empty_dataset_returns_both_leagues_without_division_errors():
    empty_shots = pd.DataFrame(
        columns=["competition_name", "shot_outcome", "match_id"]
    )

    metrics = prepare_league_metrics(empty_shots)

    assert metrics["league"].tolist() == ["NWSL", "WSL"]
    assert (metrics[["shots", "goals", "matches"]] == 0).all().all()
    assert (metrics[["conversion_rate", "shots_per_match"]] == 0).all().all()


def test_unknown_competition_is_rejected():
    shots = pd.DataFrame(
        [
            {
                "competition_name": "Other League",
                "shot_outcome": "Goal",
                "match_id": 1,
            }
        ]
    )

    with pytest.raises(ValueError, match="Unsupported competition"):
        prepare_league_metrics(shots)
