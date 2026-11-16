import pandas as pd
import pytest

from soccer_dashboard.play_pattern import (
    create_play_pattern_figure,
    prepare_play_pattern_data,
)


@pytest.fixture
def play_pattern_shots():
    return pd.DataFrame(
        [
            {"competition_name": "NWSL", "play_pattern": "Regular Play"},
            {"competition_name": "NWSL", "play_pattern": "Regular Play"},
            {"competition_name": "NWSL", "play_pattern": "From Corner"},
            {
                "competition_name": "FA Women's Super League",
                "play_pattern": "Regular Play",
            },
            {
                "competition_name": "FA Women's Super League",
                "play_pattern": "From Free Kick",
            },
            {
                "competition_name": "FA Women's Super League",
                "play_pattern": "From Free Kick",
            },
        ]
    )


def _league_rows(summary, league):
    return summary.loc[summary["league"].eq(league)].set_index("play_pattern")


def test_summary_calculates_league_counts_and_percentages(play_pattern_shots):
    summary, missing_count = prepare_play_pattern_data(play_pattern_shots)

    assert missing_count == 0
    nwsl = _league_rows(summary, "NWSL")
    wsl = _league_rows(summary, "WSL")
    assert nwsl.loc["Regular Play", ["shots", "total_shots"]].tolist() == [2, 3]
    assert nwsl.loc["Regular Play", "percentage"] == pytest.approx(200 / 3)
    assert nwsl.loc["From Free Kick", "shots"] == 0
    assert nwsl.loc["From Free Kick", "percentage"] == 0
    assert wsl.loc["From Free Kick", "shots"] == 2
    assert wsl.loc["From Free Kick", "percentage"] == pytest.approx(200 / 3)


def test_percentages_sum_to_one_hundred_in_each_league(play_pattern_shots):
    summary, _ = prepare_play_pattern_data(play_pattern_shots)

    assert summary.groupby("league")["percentage"].sum().to_dict() == pytest.approx(
        {"NWSL": 100, "WSL": 100}
    )


@pytest.mark.parametrize("missing_pattern", [None, pd.NA, float("nan"), "   "])
def test_missing_patterns_are_counted_and_included_in_denominator(
    missing_pattern,
):
    shots = pd.DataFrame(
        [
            {"competition_name": "NWSL", "play_pattern": "Regular Play"},
            {"competition_name": "NWSL", "play_pattern": missing_pattern},
            {
                "competition_name": "FA Women's Super League",
                "play_pattern": "Regular Play",
            },
        ]
    )

    summary, missing_count = prepare_play_pattern_data(shots)
    nwsl = _league_rows(summary, "NWSL")

    assert missing_count == 1
    assert nwsl.loc["Missing play pattern", "shots"] == 1
    assert nwsl.loc["Missing play pattern", "percentage"] == 50
    assert nwsl.loc["Regular Play", "percentage"] == 50
    assert _league_rows(summary, "WSL").loc["Missing play pattern", "shots"] == 0
    assert summary.groupby("league")["percentage"].sum().to_dict() == pytest.approx(
        {"NWSL": 100, "WSL": 100}
    )


def test_all_patterns_are_ordered_by_combined_frequency(play_pattern_shots):
    summary, _ = prepare_play_pattern_data(play_pattern_shots)
    figure = create_play_pattern_figure(play_pattern_shots)
    expected_order = ["Regular Play", "From Free Kick", "From Corner"]

    assert summary["play_pattern"].drop_duplicates().tolist() == expected_order
    assert list(figure.layout.yaxis.categoryarray) == expected_order
    labels = list(figure.layout.annotations[: len(expected_order)])
    assert [annotation.text for annotation in labels] == expected_order
    assert all(annotation.x == 0.5 for annotation in labels)
    assert all(annotation.xref == "paper" for annotation in labels)
    assert all(annotation.yref == "y" for annotation in labels)


def test_chart_uses_a_symmetric_percentage_scale_and_complete_hover_data(
    play_pattern_shots,
):
    figure = create_play_pattern_figure(play_pattern_shots)
    nws = next(trace for trace in figure.data if trace.name == "NWSL")
    wsl = next(trace for trace in figure.data if trace.name == "WSL")

    assert list(nws.x) == pytest.approx([-200 / 3, 0, -100 / 3])
    assert list(wsl.x) == pytest.approx([100 / 3, 200 / 3, 0])
    assert all(value <= 0 for value in nws.x)
    assert all(value >= 0 for value in wsl.x)
    assert figure.layout.xaxis.range == (-figure.layout.xaxis3.range[1], 0)
    assert figure.layout.xaxis3.range[0] == 0
    assert figure.layout.xaxis.range[0] == -figure.layout.xaxis3.range[1]
    assert list(figure.layout.xaxis.ticktext) == list(figure.layout.xaxis3.ticktext)
    assert list(figure.layout.xaxis.tickvals) == [
        -value for value in figure.layout.xaxis3.tickvals
    ]
    assert figure.layout.xaxis2.visible is False
    assert figure.layout.xaxis2.showgrid is False
    assert figure.layout.xaxis.domain[1] == pytest.approx(
        figure.layout.xaxis2.domain[0]
    )
    assert figure.layout.xaxis2.domain[1] == pytest.approx(
        figure.layout.xaxis3.domain[0]
    )
    assert figure.layout.xaxis2.domain[1] > figure.layout.xaxis2.domain[0]
    assert figure.layout.xaxis2.domain[1] - figure.layout.xaxis2.domain[0] == (
        pytest.approx(0.4)
    )
    assert figure.layout.xaxis.domain[1] - figure.layout.xaxis.domain[0] == (
        pytest.approx(0.3)
    )
    assert figure.layout.xaxis3.domain[1] - figure.layout.xaxis3.domain[0] == (
        pytest.approx(0.3)
    )
    assert nws.xaxis == "x"
    assert wsl.xaxis == "x3"
    assert "League: NWSL" in nws.hovertemplate
    assert "Play pattern: %{y}" in nws.hovertemplate
    assert "Shots: %{customdata[0]:,.0f}" in nws.hovertemplate
    assert "Share of league shots: %{customdata[1]:.2f}%" in nws.hovertemplate
    assert figure.layout.height == max(
        340, 36 * len(nws.y) + 128
    )


def test_missing_league_categories_are_present_as_zero_percent(play_pattern_shots):
    figure = create_play_pattern_figure(play_pattern_shots)
    nws = next(trace for trace in figure.data if trace.name == "NWSL")
    wsl = next(trace for trace in figure.data if trace.name == "WSL")

    assert list(nws.y) == list(wsl.y)
    assert "From Free Kick" in list(nws.y)
    assert nws.x[list(nws.y).index("From Free Kick")] == 0
    assert wsl.x[list(wsl.y).index("From Corner")] == 0


def test_tick_scale_stays_symmetric_above_eighty_percent():
    shots = pd.DataFrame(
        [
            {"competition_name": "NWSL", "play_pattern": "Regular Play"}
            for _ in range(17)
        ]
        + [
            {"competition_name": "NWSL", "play_pattern": "From Corner"}
            for _ in range(3)
        ]
        + [
            {
                "competition_name": "FA Women's Super League",
                "play_pattern": "Regular Play",
            }
        ]
    )

    figure = create_play_pattern_figure(shots)
    tick_values = list(figure.layout.xaxis3.tickvals)

    assert tick_values == sorted(tick_values)
    assert list(figure.layout.xaxis.tickvals) == [-value for value in tick_values]
    assert figure.layout.xaxis.range == (-100, 0)
    assert figure.layout.xaxis3.range == (0, 100)
