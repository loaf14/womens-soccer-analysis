import pandas as pd
import pytest

from soccer_dashboard.sankey import create_sankey_figure


@pytest.fixture
def sankey_shots():
    return pd.DataFrame(
        [
            {
                "play_pattern": "Regular Play",
                "in_box": True,
                "shot_body_part": "Right Foot",
                "shot_outcome": "Goal",
            },
            {
                "play_pattern": "Regular Play",
                "in_box": True,
                "shot_body_part": "Right Foot",
                "shot_outcome": "Saved",
            },
            {
                "play_pattern": "Regular Play",
                "in_box": False,
                "shot_body_part": "Left Foot",
                "shot_outcome": "Blocked",
            },
            {
                "play_pattern": "From Corner",
                "in_box": True,
                "shot_body_part": "Head",
                "shot_outcome": "Goal",
            },
            {
                "play_pattern": None,
                "in_box": None,
                "shot_body_part": None,
                "shot_outcome": None,
            },
        ]
    )


def _sankey(figure):
    return figure.data[0]


def _node_index(trace, stage, label):
    labels = list(trace.node.label)
    stages = list(trace.node.customdata)
    matches = [
        index
        for index, (node_stage, node_label) in enumerate(
            zip(stages, labels, strict=True)
        )
        if node_stage == stage and node_label == label
    ]
    assert len(matches) == 1
    return matches[0]


def _link_count(trace, source_stage, source_label, target_stage, target_label):
    source_index = _node_index(trace, source_stage, source_label)
    target_index = _node_index(trace, target_stage, target_label)
    return sum(
        value
        for source, target, value in zip(
            trace.link.source,
            trace.link.target,
            trace.link.value,
            strict=True,
        )
        if source == source_index and target == target_index
    )


def test_sankey_aggregates_categories_and_uses_source_category_percentages(
    sankey_shots,
):
    trace = _sankey(create_sankey_figure(sankey_shots))
    regular_play = _node_index(trace, "Play pattern", "Regular Play")
    inside_box = _node_index(trace, "Shot location", "Inside box")
    link_index = next(
        index
        for index, (source, target) in enumerate(
            zip(trace.link.source, trace.link.target, strict=True)
        )
        if source == regular_play and target == inside_box
    )

    assert (
        _link_count(
            trace,
            "Play pattern",
            "Regular Play",
            "Shot location",
            "Inside box",
        )
        == 2
    )
    assert trace.link.value[link_index] == 2
    assert list(trace.link.customdata[link_index]) == [
        "Regular Play",
        "Inside box",
        3,
        pytest.approx(200 / 3),
    ]
    assert "Share of source category" in trace.link.hovertemplate
    assert "Source category total" in trace.link.hovertemplate


def test_each_shot_contributes_once_to_every_adjacent_stage(sankey_shots):
    trace = _sankey(create_sankey_figure(sankey_shots))
    transitions = [
        ("Play pattern", "Shot location"),
        ("Shot location", "Body part"),
        ("Body part", "Outcome"),
    ]

    for source_stage, target_stage in transitions:
        stage_total = sum(
            value
            for source, target, value in zip(
                trace.link.source,
                trace.link.target,
                trace.link.value,
                strict=True,
            )
            if trace.node.customdata[source] == source_stage
            and trace.node.customdata[target] == target_stage
        )
        assert stage_total == len(sankey_shots)

    assert sum(trace.link.value) == 3 * len(sankey_shots)


def test_missing_categories_are_explicit_and_missing_outcomes_are_unknown(
    sankey_shots,
):
    trace = _sankey(create_sankey_figure(sankey_shots))
    labels = list(trace.node.label)
    stages = list(trace.node.customdata)

    assert sum(label == "Unknown" for label in labels) == 4
    assert ("Outcome", "Unknown") in list(zip(stages, labels, strict=True))
    assert ("Outcome", "Non-goal") in list(zip(stages, labels, strict=True))
    assert ("Outcome", "Goal") in list(zip(stages, labels, strict=True))
    assert _link_count(trace, "Body part", "Unknown", "Outcome", "Unknown") == 1
    assert _link_count(trace, "Body part", "Unknown", "Outcome", "Non-goal") == 0


def test_missing_and_blank_outcomes_are_unknown_not_non_goals():
    shots = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 7,
            "in_box": [True] * 7,
            "shot_body_part": ["Right Foot"] * 7,
            "shot_outcome": [
                None,
                pd.NA,
                "",
                "   ",
                "Goal",
                "Saved",
                "Unknown",
            ],
        }
    )
    trace = _sankey(create_sankey_figure(shots))
    outcome_nodes = [
        label
        for stage, label in zip(
            trace.node.customdata, trace.node.label, strict=True
        )
        if stage == "Outcome"
    ]

    assert outcome_nodes == ["Goal", "Non-goal", "Unknown"]
    assert _link_count(trace, "Body part", "Right Foot", "Outcome", "Unknown") == 4
    assert _link_count(trace, "Body part", "Right Foot", "Outcome", "Goal") == 1
    assert _link_count(trace, "Body part", "Right Foot", "Outcome", "Non-goal") == 2

    for source_stage, target_stage in [
        ("Play pattern", "Shot location"),
        ("Shot location", "Body part"),
        ("Body part", "Outcome"),
    ]:
        assert sum(
            value
            for source, target, value in zip(
                trace.link.source,
                trace.link.target,
                trace.link.value,
                strict=True,
            )
            if trace.node.customdata[source] == source_stage
            and trace.node.customdata[target] == target_stage
        ) == len(shots)
    assert sum(trace.link.value) == 3 * len(shots)


def test_rare_body_parts_are_grouped_as_other():
    shots = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 201,
            "in_box": [True] * 201,
            "shot_body_part": ["Right Foot"] * 200 + ["Toe"],
            "shot_outcome": ["Saved"] * 201,
        }
    )

    labels = list(_sankey(create_sankey_figure(shots)).node.label)

    assert "Toe" not in labels
    assert "Other" in labels


def test_empty_dataset_returns_empty_sankey_trace():
    shots = pd.DataFrame(
        columns=["play_pattern", "in_box", "shot_body_part", "shot_outcome"]
    )

    trace = _sankey(create_sankey_figure(shots))

    assert list(trace.node.label) == []
    assert list(trace.link.source) == []
    assert list(trace.link.target) == []
    assert list(trace.link.value) == []
