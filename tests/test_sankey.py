import pandas as pd
import pytest

from soccer_dashboard.sankey import (
    create_sankey_figure,
    sankey_category_orders,
)


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


def _transition_total(trace, source_stage, target_stage):
    return sum(
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
        if source == inside_box and target == regular_play
    )

    assert (
        _link_count(
            trace,
            "Shot location",
            "Inside box",
            "Play pattern",
            "Regular Play",
        )
        == 2
    )
    assert trace.link.value[link_index] == 2
    assert list(trace.link.customdata[link_index]) == [
        "Inside box",
        "Regular Play",
        3,
        pytest.approx(200 / 3),
    ]
    assert "Share of source category" in trace.link.hovertemplate
    assert "Source category total" in trace.link.hovertemplate


def test_default_mode_includes_all_shots_and_four_stages(sankey_shots):
    trace = _sankey(create_sankey_figure(sankey_shots))

    assert list(dict.fromkeys(trace.node.customdata)) == [
        "Shot location",
        "Play pattern",
        "Body part",
        "Outcome",
    ]
    assert len(set(trace.node.customdata)) == 4
    assert [
        _transition_total(trace, source, target)
        for source, target in [
            ("Shot location", "Play pattern"),
            ("Play pattern", "Body part"),
            ("Body part", "Outcome"),
        ]
    ] == [len(sankey_shots)] * 3

    assert sum(trace.link.value) == 3 * len(sankey_shots)
    assert _link_count(
        trace,
        "Play pattern",
        "Regular Play",
        "Body part",
        "Right Foot",
    ) == 2

    expected_node_colors = {
        "Shot location": "#78a6d8",
        "Play pattern": "#62b88a",
        "Body part": "#e0ad68",
        "Outcome": "#d78273",
    }
    assert list(trace.node.color) == [
        expected_node_colors[stage] for stage in trace.node.customdata
    ]

    expected_link_colors = {
        ("Shot location", "Play pattern"): "rgba(120,166,216,0.24)",
        ("Play pattern", "Body part"): "rgba(98,184,138,0.24)",
        ("Body part", "Outcome"): "rgba(224,173,104,0.24)",
    }
    assert {
        (
            trace.node.customdata[source],
            trace.node.customdata[target],
        ): color
        for source, target, color in zip(
            trace.link.source,
            trace.link.target,
            trace.link.color,
            strict=True,
        )
    } == expected_link_colors


def test_goals_only_mode_filters_shots_and_uses_three_stages(sankey_shots):
    trace = _sankey(create_sankey_figure(sankey_shots, mode="goals"))

    assert list(dict.fromkeys(trace.node.customdata)) == [
        "Shot location",
        "Play pattern",
        "Body part",
    ]
    assert len(set(trace.node.customdata)) == 3
    assert _transition_total(trace, "Shot location", "Play pattern") == 2
    assert _transition_total(trace, "Play pattern", "Body part") == 2
    assert sum(trace.link.value) == 2 * 2
    assert _link_count(
        trace,
        "Shot location",
        "Inside box",
        "Play pattern",
        "Regular Play",
    ) == 1


def test_goals_only_recalculates_percentages_from_selected_goals(sankey_shots):
    trace = _sankey(create_sankey_figure(sankey_shots, mode="goals"))
    source = _node_index(trace, "Shot location", "Inside box")
    target = _node_index(trace, "Play pattern", "Regular Play")
    link_index = next(
        index
        for index, (link_source, link_target) in enumerate(
            zip(trace.link.source, trace.link.target, strict=True)
        )
        if link_source == source and link_target == target
    )

    assert list(trace.link.customdata[link_index]) == [
        "Inside box",
        "Regular Play",
        2,
        pytest.approx(50),
    ]


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
        ("Shot location", "Play pattern"),
        ("Play pattern", "Body part"),
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


def test_infrequent_play_patterns_remain_distinct_across_leagues_and_modes():
    shots = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 200 + ["From Kick Off"],
            "in_box": [True] * 201,
            "shot_body_part": ["Right Foot"] * 201,
            "shot_outcome": ["Saved"] * 199 + ["Goal", "Goal"],
        }
    )
    other_league = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 200 + ["From Corner"],
            "in_box": [False] * 201,
            "shot_body_part": ["Left Foot"] * 201,
            "shot_outcome": ["Saved"] * 200 + ["Goal"],
        }
    )
    league_shots = {"nwsl": shots, "wsl": other_league}
    all_orders = sankey_category_orders(league_shots)
    goal_orders = sankey_category_orders(league_shots, mode="goals")

    assert all_orders["play_pattern"] == goal_orders["play_pattern"]
    assert "From Kick Off" in all_orders["play_pattern"]
    assert "Other" not in all_orders["play_pattern"]

    for mode, orders in (("all", all_orders), ("goals", goal_orders)):
        nwsl_trace = _sankey(
            create_sankey_figure(shots, mode=mode, category_orders=orders)
        )
        wsl_trace = _sankey(
            create_sankey_figure(
                other_league, mode=mode, category_orders=orders
            )
        )

        assert _node_index(nwsl_trace, "Play pattern", "From Kick Off") >= 0
        assert "From Kick Off" not in list(wsl_trace.node.label)
        assert _link_count(
            nwsl_trace,
            "Shot location",
            "Inside box",
            "Play pattern",
            "From Kick Off",
        ) == 1
        assert _transition_total(
            nwsl_trace, "Shot location", "Play pattern"
        ) == (201 if mode == "all" else 2)


def test_shared_category_orders_align_league_categories_without_inventing_nodes():
    nwsl = pd.DataFrame(
        {
            "play_pattern": ["Regular Play", "From Corner", "Regular Play"],
            "in_box": [True, True, False],
            "shot_body_part": ["Right Foot", "Head", "Left Foot"],
            "shot_outcome": ["Goal", "Saved", "Blocked"],
        }
    )
    wsl = pd.DataFrame(
        {
            "play_pattern": ["Regular Play", "From Free Kick"],
            "in_box": [True, False],
            "shot_body_part": ["Right Foot", "Head"],
            "shot_outcome": ["Goal", "Saved"],
        }
    )
    orders = sankey_category_orders({"nwsl": nwsl, "wsl": wsl})
    nwsl_trace = _sankey(
        create_sankey_figure(nwsl, category_orders=orders)
    )
    wsl_trace = _sankey(
        create_sankey_figure(wsl, category_orders=orders)
    )

    def stage_labels(trace, stage):
        return [
            label
            for node_stage, label in zip(
                trace.node.customdata, trace.node.label, strict=True
            )
            if node_stage == stage
        ]

    nwsl_patterns = stage_labels(nwsl_trace, "Play pattern")
    wsl_patterns = stage_labels(wsl_trace, "Play pattern")
    shared_patterns = set(nwsl_patterns) & set(wsl_patterns)

    assert [label for label in nwsl_patterns if label in shared_patterns] == [
        label for label in wsl_patterns if label in shared_patterns
    ]
    assert "From Corner" in nwsl_patterns
    assert "From Corner" not in wsl_patterns
    assert "From Free Kick" not in nwsl_patterns
    assert "From Free Kick" in wsl_patterns


def test_play_pattern_order_and_positions_are_shared_across_leagues_and_modes():
    nwsl = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 8 + ["From Corner"] * 2,
            "in_box": [True] * 10,
            "shot_body_part": ["Right Foot"] * 10,
            "shot_outcome": ["Goal"] + ["Saved"] * 7 + ["Goal"] * 2,
        }
    )
    wsl = pd.DataFrame(
        {
            "play_pattern": ["Regular Play"] * 4 + ["From Corner"] * 4,
            "in_box": [False] * 8,
            "shot_body_part": ["Left Foot"] * 8,
            "shot_outcome": ["Goal"] + ["Saved"] * 3 + ["Goal"] + ["Saved"] * 3,
        }
    )
    league_shots = {"nwsl": nwsl, "wsl": wsl}
    all_orders = sankey_category_orders(league_shots)
    goal_orders = sankey_category_orders(league_shots, mode="goals")

    assert all_orders["play_pattern"] == goal_orders["play_pattern"]

    traces = [
        _sankey(
            create_sankey_figure(shots, mode=mode, category_orders=orders)
        )
        for mode, orders in (("all", all_orders), ("goals", goal_orders))
        for shots in (nwsl, wsl)
    ]
    expected_order = ["From Corner", "Regular Play"]

    for trace in traces:
        assert trace.arrangement == "fixed"
        assert len(trace.node.x) == len(trace.node.label)
        assert len(trace.node.y) == len(trace.node.label)
        pattern_nodes = [
            (label, y)
            for stage, label, y in zip(
                trace.node.customdata,
                trace.node.label,
                trace.node.y,
                strict=True,
            )
            if stage == "Play pattern"
        ]
        assert [label for label, _ in pattern_nodes] == expected_order
        assert [y for _, y in pattern_nodes] == sorted(
            y for _, y in pattern_nodes
        )

        stage_counts = {
            stage: list(trace.node.customdata).count(stage)
            for stage in set(trace.node.customdata)
        }
        plot_height = 500 - 20 - 12
        node_pad = 12
        node_scale = (
            plot_height - node_pad * (max(stage_counts.values()) - 1)
        ) / sum(trace.link.value) * (len(stage_counts) - 1)
        for stage in stage_counts:
            node_indices = [
                index
                for index, node_stage in enumerate(trace.node.customdata)
                if node_stage == stage
            ]
            node_sizes = []
            for index in node_indices:
                outgoing = sum(
                    value
                    for source, value in zip(
                        trace.link.source, trace.link.value, strict=True
                    )
                    if source == index
                )
                incoming = sum(
                    value
                    for target, value in zip(
                        trace.link.target, trace.link.value, strict=True
                    )
                    if target == index
                )
                node_sizes.append(outgoing or incoming)

            for previous, current, previous_size, current_size in zip(
                node_indices[:-1],
                node_indices[1:],
                node_sizes[:-1],
                node_sizes[1:],
                strict=True,
            ):
                center_distance = (
                    trace.node.y[current] - trace.node.y[previous]
                ) * plot_height
                minimum_distance = (
                    (previous_size + current_size) * node_scale / 2 + node_pad
                )
                assert center_distance + 1e-6 >= minimum_distance


def test_empty_dataset_returns_empty_sankey_trace():
    shots = pd.DataFrame(
        columns=["play_pattern", "in_box", "shot_body_part", "shot_outcome"]
    )

    trace = _sankey(create_sankey_figure(shots))

    assert list(trace.node.label) == []
    assert list(trace.link.source) == []
    assert list(trace.link.target) == []
    assert list(trace.link.value) == []


def test_empty_goals_only_dataset_has_no_nodes_or_links():
    shots = pd.DataFrame(
        columns=["play_pattern", "in_box", "shot_body_part", "shot_outcome"]
    )

    trace = _sankey(create_sankey_figure(shots, mode="goals"))

    assert list(trace.node.label) == []
    assert list(trace.link.source) == []
    assert list(trace.link.target) == []
    assert list(trace.link.value) == []


def test_invalid_mode_is_rejected(sankey_shots):
    with pytest.raises(ValueError, match="Sankey mode"):
        create_sankey_figure(sankey_shots, mode="other")
