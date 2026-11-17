import pandas as pd
import pytest

from soccer_dashboard.shot_conversion import (
    create_assist_conversion_figure,
    create_play_pattern_conversion_figure,
    prepare_assist_conversion_data,
    prepare_play_pattern_conversion_data,
)


@pytest.fixture
def conversion_shots():
    rows = [
        ("NWSL", "Regular Play", "Goal", True),
        ("NWSL", "Regular Play", "Saved", True),
        ("NWSL", "Regular Play", "Blocked", False),
        ("NWSL", "From Corner", "Goal", False),
        ("NWSL", "From Corner", "Saved", False),
        ("NWSL", None, "Saved", False),
        ("FA Women's Super League", "Regular Play", "Goal", True),
        ("FA Women's Super League", "Regular Play", "Saved", False),
        ("FA Women's Super League", "From Counter", "Goal", True),
        ("FA Women's Super League", "From Counter", "Saved", False),
        ("FA Women's Super League", "From Free Kick", "Goal", True),
    ]
    shots = pd.DataFrame(
        [
            {
                "competition_name": league,
                "play_pattern": pattern,
                "shot_outcome": outcome,
                "shot_key_pass_id": f"key-{index}" if assisted else None,
                "assisted": assisted,
            }
            for index, (league, pattern, outcome, assisted) in enumerate(rows)
        ]
    )
    shots["shot_type"] = "Open Play"
    shots.loc[0, "shot_type"] = "Penalty"
    return shots


def _row(summary, league, category, column):
    return summary.loc[
        summary["league"].eq(league) & summary[column].eq(category)
    ].iloc[0]


def test_play_pattern_summary_sorts_all_categories_by_combined_shots(
    conversion_shots,
):
    summary = prepare_play_pattern_conversion_data(conversion_shots)

    assert summary["shots"].sum() == len(conversion_shots)
    assert summary["play_pattern"].drop_duplicates().tolist() == [
        "Regular Play",
        "From Corner",
        "From Counter",
        "From Free Kick",
        "Unknown",
    ]
    regular_nwsl = _row(summary, "NWSL", "Regular Play", "play_pattern")
    regular_wsl = _row(
        summary, "WSL", "Regular Play", "play_pattern"
    )
    assert (regular_nwsl["shots"], regular_nwsl["goals"]) == (3, 1)
    assert regular_nwsl["conversion_rate"] == pytest.approx(100 / 3)
    assert (regular_wsl["shots"], regular_wsl["goals"]) == (2, 1)
    assert regular_wsl["conversion_rate"] == pytest.approx(50)


def test_missing_league_pattern_has_no_fabricated_conversion_rate(
    conversion_shots,
):
    summary = prepare_play_pattern_conversion_data(conversion_shots)
    missing_group = _row(summary, "WSL", "From Corner", "play_pattern")

    assert missing_group["shots"] == 0
    assert missing_group["goals"] == 0
    assert pd.isna(missing_group["conversion_rate"])
    assert "Unknown" in summary["play_pattern"].unique()


def test_play_pattern_dumbbell_uses_league_colors_counts_and_hover(
    conversion_shots,
):
    figure = create_play_pattern_conversion_figure(conversion_shots)
    nwsl = next(trace for trace in figure.data if trace.name == "NWSL")
    wsl = next(trace for trace in figure.data if trace.name == "WSL")
    connectors = [
        trace
        for trace in figure.data
        if trace.mode == "lines"
    ]

    assert nwsl.marker.color == "#e7a45e"
    assert wsl.marker.color == "#8db4ff"
    assert len(connectors) == 1
    assert set(nwsl.y) == {
        "Regular Play",
        "From Corner",
        "Unknown",
    }
    assert "From Corner" not in set(wsl.y)
    assert "From Free Kick" in set(wsl.y)
    assert "Unknown" in figure.layout.yaxis.categoryarray
    assert figure.layout.yaxis.categoryarray[-1] == "Regular Play"
    assert figure.layout.xaxis.range[0] == 0
    assert figure.layout.xaxis.ticksuffix == "%"
    assert "Play pattern: %{y}" in nwsl.hovertemplate
    assert "Shots: %{customdata[0]:,.0f}" in nwsl.hovertemplate
    assert "Goals: %{customdata[1]:,.0f}" in nwsl.hovertemplate
    assert "Conversion rate: %{customdata[2]:.1f}%" in nwsl.hovertemplate
    assert figure.layout.legend.orientation == "h"


def test_assist_summary_validates_categories_and_shot_shares(conversion_shots):
    summary = prepare_assist_conversion_data(conversion_shots)
    nwsl_assisted = _row(summary, "NWSL", "Assisted", "assist_status")
    nwsl_unassisted = _row(summary, "NWSL", "Unassisted", "assist_status")
    wsl_assisted = _row(summary, "WSL", "Assisted", "assist_status")

    assert (nwsl_assisted["shots"], nwsl_assisted["goals"]) == (2, 1)
    assert nwsl_assisted["conversion_rate"] == pytest.approx(50)
    assert nwsl_assisted["league_shot_share"] == pytest.approx(2 / 6 * 100)
    assert (nwsl_unassisted["shots"], nwsl_unassisted["goals"]) == (4, 1)
    assert nwsl_unassisted["league_shot_share"] == pytest.approx(4 / 6 * 100)
    assert (wsl_assisted["shots"], wsl_assisted["goals"]) == (3, 3)
    assert wsl_assisted["conversion_rate"] == pytest.approx(100)
    assert summary["shots"].sum() == len(conversion_shots)
    assert summary.groupby("league")["league_shot_share"].sum().to_dict() == {
        "NWSL": pytest.approx(100),
        "WSL": pytest.approx(100),
    }


def test_assist_chart_has_grouped_league_bars_and_complete_hover(
    conversion_shots,
):
    figure = create_assist_conversion_figure(conversion_shots)
    traces = {trace.name: trace for trace in figure.data}
    nws = traces["NWSL"]
    wsl = traces["WSL"]

    assert figure.layout.barmode == "group"
    assert list(nws.x) == ["Assisted", "Unassisted"]
    assert list(nws.y) == [50, 25]
    assert list(wsl.y) == [100, 0]
    assert nws.marker.color == "#e7a45e"
    assert wsl.marker.color == "#8db4ff"
    assert list(nws.text) == ["50.0%", "25.0%"]
    assert "Assist status: %{x}" in nws.hovertemplate
    assert "Shots: %{customdata[0]:,.0f}" in nws.hovertemplate
    assert "Goals: %{customdata[1]:,.0f}" in nws.hovertemplate
    assert "Conversion rate: %{customdata[2]:.1f}%" in nws.hovertemplate
    assert "Share of league shots: %{customdata[3]:.1f}%" in nws.hovertemplate
    assert figure.layout.yaxis.range[0] == 0
    assert figure.layout.yaxis.ticksuffix == "%"


def test_assisted_flag_must_match_key_pass_presence(conversion_shots):
    inconsistent = conversion_shots.copy()
    inconsistent.loc[0, "assisted"] = False

    with pytest.raises(ValueError, match="must match shot_key_pass_id"):
        prepare_assist_conversion_data(inconsistent)


def test_missing_assist_columns_are_rejected():
    shots = pd.DataFrame(
        columns=["competition_name", "shot_outcome", "shot_key_pass_id"]
    )

    with pytest.raises(ValueError, match="missing assist conversion columns"):
        prepare_assist_conversion_data(shots)


def test_empty_groups_have_no_conversion_rates_or_shot_shares():
    shots = pd.DataFrame(
        {
            "competition_name": pd.Series(dtype="object"),
            "shot_outcome": pd.Series(dtype="object"),
            "shot_key_pass_id": pd.Series(dtype="object"),
            "assisted": pd.Series(dtype="bool"),
            "play_pattern": pd.Series(dtype="object"),
        }
    )

    pattern_summary = prepare_play_pattern_conversion_data(shots)
    assist_summary = prepare_assist_conversion_data(shots)
    pattern_figure = create_play_pattern_conversion_figure(shots)
    assist_figure = create_assist_conversion_figure(shots)

    assert pattern_summary.empty
    assert assist_summary["shots"].sum() == 0
    assert assist_summary["conversion_rate"].isna().all()
    assert assist_summary["league_shot_share"].isna().all()
    assert pattern_figure.layout.xaxis.range[0] == 0
    assert assist_figure.layout.yaxis.range[0] == 0


def test_reference_assist_counts_and_rates_are_calculated_from_shots():
    reference = [
        ("NWSL", True, 2426, 217),
        ("NWSL", False, 1104, 117),
        ("FA Women's Super League", True, 2443, 280),
        ("FA Women's Super League", False, 1086, 140),
    ]
    rows = []
    for league, assisted, count, goals in reference:
        rows.extend(
            {
                "competition_name": league,
                "shot_outcome": "Goal" if index < goals else "Saved",
                "shot_key_pass_id": f"{league}-{assisted}-{index}" if assisted else None,
                "assisted": assisted,
            }
            for index in range(count)
        )

    summary = prepare_assist_conversion_data(pd.DataFrame(rows))
    expected = {
        ("NWSL", "Assisted"): (2426, 217, 8.9448, 68.7252),
        ("NWSL", "Unassisted"): (1104, 117, 10.5978, 31.2748),
        ("WSL", "Assisted"): (2443, 280, 11.4613, 69.2264),
        ("WSL", "Unassisted"): (1086, 140, 12.8913, 30.7736),
    }
    for (league, status), values in expected.items():
        row = _row(summary, league, status, "assist_status")
        assert (row["shots"], row["goals"]) == values[:2]
        assert row["conversion_rate"] == pytest.approx(values[2], abs=0.0001)
        assert row["league_shot_share"] == pytest.approx(values[3], abs=0.0001)
