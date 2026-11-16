import pandas as pd
import pytest

from soccer_dashboard.shot_conversion import (
    create_conversion_figure,
    prepare_conversion_data,
)


@pytest.fixture
def conversion_shots():
    return pd.DataFrame(
        [
            {
                "competition_name": "NWSL",
                "in_box": True,
                "shot_outcome": "Goal",
            },
            {
                "competition_name": "NWSL",
                "in_box": True,
                "shot_outcome": "Saved",
            },
            {
                "competition_name": "NWSL",
                "in_box": False,
                "shot_outcome": "Blocked",
            },
            {
                "competition_name": "NWSL",
                "in_box": False,
                "shot_outcome": "Goal",
            },
            {
                "competition_name": "FA Women's Super League",
                "in_box": True,
                "shot_outcome": "Goal",
            },
            {
                "competition_name": "FA Women's Super League",
                "in_box": False,
                "shot_outcome": "Saved",
            },
            {
                "competition_name": "NWSL",
                "in_box": None,
                "shot_outcome": "Goal",
            },
        ]
    )


def _row(summary, league, location):
    return summary.loc[
        summary["league"].eq(league) & summary["location"].eq(location)
    ].iloc[0]


def test_conversion_summary_counts_goals_shots_and_missing_locations(
    conversion_shots,
):
    summary, missing_locations = prepare_conversion_data(conversion_shots)

    assert missing_locations == 1
    assert tuple(_row(summary, "NWSL", "Inside box")[["shots", "goals"]]) == (2, 1)
    assert _row(summary, "NWSL", "Inside box")["conversion_rate"] == pytest.approx(50)
    assert tuple(_row(summary, "NWSL", "Outside box")[["shots", "goals"]]) == (2, 1)
    assert _row(summary, "NWSL", "Outside box")["conversion_rate"] == pytest.approx(50)
    assert tuple(_row(summary, "WSL", "Inside box")[["shots", "goals"]]) == (1, 1)
    assert _row(summary, "WSL", "Inside box")["conversion_rate"] == pytest.approx(100)
    assert tuple(_row(summary, "WSL", "Outside box")[["shots", "goals"]]) == (1, 0)
    assert _row(summary, "WSL", "Outside box")["conversion_rate"] == pytest.approx(0)
    assert summary["shots"].sum() == 6
    assert summary["goals"].sum() == 3


def test_chart_marker_positions_and_hover_match_group_calculations(conversion_shots):
    figure = create_conversion_figure(conversion_shots)
    nws = next(trace for trace in figure.data if trace.name == "NWSL")
    wsl = next(trace for trace in figure.data if trace.name == "WSL")

    assert list(nws.x) == [50, 50]
    assert list(nws.y) == ["Inside box", "Outside box"]
    assert list(nws.text) == ["50.0%", "50.0%"]
    assert list(nws.customdata[0]) == [2, 1, 50]
    assert list(wsl.x) == [100, 0]
    assert list(wsl.text) == ["100.0%", "0.0%"]
    assert list(wsl.customdata[0]) == [1, 1, 100]
    assert "Shots: %{customdata[0]:,.0f}" in nws.hovertemplate
    assert "Goals: %{customdata[1]:,.0f}" in nws.hovertemplate
    assert "Conversion rate: %{customdata[2]:.1f}%" in nws.hovertemplate
    assert figure.layout.xaxis.range[0] == 0
    assert figure.layout.xaxis.ticksuffix == "%"
    assert figure.layout.height == 260
    assert figure.layout.legend.orientation == "h"
    assert figure.layout.legend.x == 0
    assert figure.layout.legend.xanchor == "left"
    assert [trace.name for trace in figure.data if trace.mode == "lines"] == [
        "Inside box connector",
        "Outside box connector",
    ]


def test_zero_shot_groups_have_no_rate_or_marker():
    shots = pd.DataFrame(
        [
            {
                "competition_name": "NWSL",
                "in_box": True,
                "shot_outcome": "Saved",
            }
        ]
    )
    summary, missing_locations = prepare_conversion_data(shots)
    figure = create_conversion_figure(shots)
    wsl = next(trace for trace in figure.data if trace.name == "WSL")

    assert missing_locations == 0
    assert _row(summary, "NWSL", "Inside box")["conversion_rate"] == 0
    assert pd.isna(_row(summary, "NWSL", "Outside box")["conversion_rate"])
    assert pd.isna(_row(summary, "WSL", "Inside box")["conversion_rate"])
    assert len(wsl.x) == 0
    assert {annotation.text for annotation in figure.layout.annotations} == {
        "NWSL: no shots",
        "WSL: no shots",
    }


@pytest.mark.parametrize("missing_location", [None, pd.NA, float("nan")])
def test_missing_in_box_values_are_excluded_and_counted(missing_location):
    shots = pd.DataFrame(
        [
            {
                "competition_name": "NWSL",
                "in_box": missing_location,
                "shot_outcome": "Goal",
            }
        ]
    )

    summary, missing_locations = prepare_conversion_data(shots)

    assert missing_locations == 1
    assert summary["shots"].sum() == 0
    assert summary["goals"].sum() == 0
    assert summary["conversion_rate"].isna().all()


def test_empty_shots_keep_zero_groups_without_inventing_rates():
    shots = pd.DataFrame(
        columns=["competition_name", "in_box", "shot_outcome"]
    )
    summary, missing_locations = prepare_conversion_data(shots)
    figure = create_conversion_figure(shots)

    assert missing_locations == 0
    assert len(summary) == 4
    assert summary["shots"].sum() == 0
    assert summary["goals"].sum() == 0
    assert summary["conversion_rate"].isna().all()
    assert figure.layout.xaxis.range[0] == 0
    assert len(figure.layout.annotations) == 4


def test_unknown_competition_is_rejected():
    shots = pd.DataFrame(
        [{"competition_name": "Other League", "in_box": True, "shot_outcome": "Goal"}]
    )

    with pytest.raises(ValueError, match="Unsupported competition"):
        prepare_conversion_data(shots)
