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


def _link_count(trace, source_label, target_label):
    labels = list(trace.node.label)
    source_index = labels.index(source_label)
    target_index = labels.index(target_label)
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
    labels = list(trace.node.label)
    regular_play = labels.index("Regular Play")
    inside_box = labels.index("Inside box")
    link_index = next(
        index
        for index, (source, target) in enumerate(
            zip(trace.link.source, trace.link.target, strict=True)
        )
        if source == regular_play and target == inside_box
    )

    assert _link_count(trace, "Regular Play", "Inside box") == 2
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
    categories = {
        "play_pattern": sankey_shots["play_pattern"].fillna("Unknown").nunique(),
        "shot_location": 3,
        "body_part": sankey_shots["shot_body_part"].fillna("Unknown").nunique(),
        "outcome": 2,
    }
    stage_starts = {}
    offset = 0
    for column, count in categories.items():
        stage_starts[column] = (offset, offset + count)
        offset += count

    for source_column, target_column in zip(
        list(categories)[:-1], list(categories)[1:], strict=True
    ):
        start, end = stage_starts[source_column]
        stage_total = sum(
            value
            for source, value in zip(
                trace.link.source, trace.link.value, strict=True
            )
            if start <= source < end
        )
        assert stage_total == len(sankey_shots)

    assert sum(trace.link.value) == 3 * len(sankey_shots)


def test_missing_categories_are_explicit_and_missing_outcomes_are_non_goals(
    sankey_shots,
):
    trace = _sankey(create_sankey_figure(sankey_shots))
    labels = list(trace.node.label)
    unknown_indices = [index for index, label in enumerate(labels) if label == "Unknown"]

    assert labels.count("Unknown") == 3
    assert "Non-goal" in labels
    assert "Goal" in labels
    assert sum(
        value
        for source, target, value in zip(
            trace.link.source,
            trace.link.target,
            trace.link.value,
            strict=True,
        )
        if source == unknown_indices[0] and target == unknown_indices[1]
    ) == 1
    assert sum(
        value
        for source, target, value in zip(
            trace.link.source,
            trace.link.target,
            trace.link.value,
            strict=True,
        )
        if source == unknown_indices[2] and target == labels.index("Non-goal")
    ) == 1


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
