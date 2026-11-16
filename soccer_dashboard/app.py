"""Dash application for women's soccer shot analysis."""

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from soccer_dashboard.sankey import create_sankey_figure
from soccer_dashboard.shot_map import create_shot_map, prepare_shot_coordinates
from soccer_dashboard.shot_conversion import (
    create_conversion_figure,
    prepare_conversion_data,
)
from soccer_dashboard.storage import load_local_shots

COMPETITION_VALUES = {
    "NWSL": "nwsl",
    "FA Women's Super League": "wsl",
}


def _competition_options(shots: pd.DataFrame) -> list[dict[str, str]]:
    available = set(shots["competition_name"].dropna().unique())
    options = [{"label": "Both leagues", "value": "all"}]
    options.extend(
        {
            "label": "WSL" if name == "FA Women's Super League" else name,
            "value": COMPETITION_VALUES[name],
        }
        for name in COMPETITION_VALUES
        if name in available
    )
    return options


def _filter(label: str, component_id: str, options: list[dict[str, str]]) -> html.Div:
    return html.Div(
        [
            html.Label(label, htmlFor=component_id, className="filter-label"),
            dcc.Dropdown(
                id=component_id,
                options=options,
                value=options[0]["value"],
                clearable=False,
                searchable=True,
                className="dashboard-dropdown",
            ),
        ],
        className="filter-control",
    )


def _metric_card(label: str, value: str, detail: str, card_id: str) -> html.Article:
    return html.Article(
        [
            html.P(label, className="metric-label"),
            html.P(value, id=card_id, className="metric-value"),
            html.P(detail, className="metric-detail"),
        ],
        className="metric-card",
    )


def _chart_placeholder(
    heading: str,
    subtitle: str,
    component_id: str,
    phase: str,
    class_name: str = "",
) -> html.Section:
    return html.Section(
        [
            html.Div(
                [
                    html.H2(heading, className="section-title"),
                    html.P(subtitle, className="section-subtitle"),
                ],
                className="visualization-heading",
            ),
            html.Div(
                [
                    html.Span(phase, className="placeholder-index"),
                    html.P("Visualization coming next", className="placeholder-title"),
                    html.P(
                        "This space is reserved for the interactive analysis.",
                        className="placeholder-copy",
                    ),
                ],
                id=component_id,
                className="visualization-placeholder",
                role="img",
                **{"aria-label": f"{heading} visualization placeholder"},
            ),
        ],
        className=f"visualization-card {class_name}".strip(),
    )


def create_app(shots: pd.DataFrame) -> Dash:
    """Create the dashboard using an already loaded shot dataset."""
    goals = int(shots["shot_outcome"].eq("Goal").sum())
    total_shots = len(shots)
    conversion_rate = goals / total_shots * 100 if total_shots else 0
    matches = int(shots.groupby(["competition_name", "match_id"]).ngroups)
    conversion_figure = create_conversion_figure(shots)
    _, conversion_missing_locations = prepare_conversion_data(shots)
    map_shots = prepare_shot_coordinates(shots)
    shots_outside_half = int(
        (
            map_shots["has_valid_location"]
            & ~map_shots["in_attacking_half"]
        ).sum()
    )
    shots_without_location = int((~map_shots["has_valid_location"]).sum())
    map_coverage_note = (
        "Half-pitch view: StatsBomb x=60–120, attacking toward the goal. "
        f"Valid shots outside the attacking half (x < 60): {shots_outside_half:,}. "
        "These are not plotted in this half-pitch view. "
        f"Shot locations unavailable or invalid: {shots_without_location:,}."
    )

    seasons = (
        shots[["competition_name", "season_name"]]
        .drop_duplicates()
        .sort_values(["competition_name", "season_name"])
    )
    season_summary = html.Div(
        [
            html.Span(
                f"{row.competition_name} · {row.season_name}",
                className=(
                    "season-chip "
                    f"{COMPETITION_VALUES.get(row.competition_name, '')}"
                ).strip(),
            )
            for row in seasons.itertuples(index=False)
        ],
        className="season-line",
    )

    app = Dash(
        __name__,
        title="The Anatomy of a Goal",
        assets_folder=str(Path(__file__).resolve().parent.parent / "assets"),
    )
    app.layout = html.Main(
        [
            html.Header(
                [
                    html.Div(
                        [
                            html.P(
                                "WOMEN'S SOCCER  /  SHOT ANALYSIS",
                                className="eyebrow",
                            ),
                            html.H1("The Anatomy of a Goal"),
                            html.P(
                                "How do shot selection and goal-scoring efficiency "
                                "differ between the NWSL and FA Women's Super League?",
                                className="hero-subtitle",
                            ),
                            season_summary,
                        ],
                        className="hero-copy",
                    ),
                    html.A(
                        [
                            html.Span("DATA SOURCE", className="source-label"),
                            html.Span("StatsBomb Open Data", className="source-name"),
                            html.Span("↗", className="source-arrow"),
                        ],
                        href="https://github.com/statsbomb/open-data",
                        target="_blank",
                        rel="noreferrer",
                        className="source-badge",
                    ),
                ],
                className="hero",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Explore the dataset", className="section-title"),
                            html.P(
                                "Choose a lens for the analysis. Filters will be "
                                "interactive in a later phase.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Div(
                        [
                            _filter(
                                "Competition",
                                "competition-filter",
                                _competition_options(shots),
                            ),
                            _filter(
                                "Team",
                                "team-filter",
                                [
                                    {"label": "All teams", "value": "all"},
                                    *[
                                        {"label": team, "value": team}
                                        for team in sorted(
                                            shots["team"].dropna().unique()
                                        )
                                    ],
                                ],
                            ),
                            _filter(
                                "Shot outcome",
                                "shot-outcome-filter",
                                [
                                    {"label": "All outcomes", "value": "all"},
                                    *[
                                        {"label": outcome, "value": outcome}
                                        for outcome in sorted(
                                            shots["shot_outcome"].dropna().unique()
                                        )
                                    ],
                                ],
                            ),
                            _filter(
                                "Shot type",
                                "shot-type-filter",
                                [
                                    {"label": "All shot types", "value": "all"},
                                    *[
                                        {"label": shot_type, "value": shot_type}
                                        for shot_type in sorted(
                                            shots["shot_type"].dropna().unique()
                                        )
                                    ],
                                ],
                            ),
                        ],
                        className="filter-grid",
                    ),
                ],
                className="dashboard-section filter-section",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            _metric_card(
                                "TOTAL SHOTS",
                                f"{total_shots:,}",
                                "Attempts across both competitions",
                                "total-shot-count",
                            ),
                            _metric_card(
                                "TOTAL GOALS",
                                f"{goals:,}",
                                "Shots recorded as goals",
                                "total-goal-count",
                            ),
                            _metric_card(
                                "SHOT CONVERSION",
                                f"{conversion_rate:.1f}%",
                                "Goals ÷ total shots",
                                "shot-conversion-rate",
                            ),
                            _metric_card(
                                "MATCHES",
                                f"{matches:,}",
                                "Unique competition and match pairs",
                                "match-count",
                            ),
                        ],
                        className="metric-grid",
                    )
                ],
                className="dashboard-section metrics-section",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.P("01  /  SHOT LOCATION", className="section-kicker"),
                            html.H2("Where shots become goals", className="section-title"),
                            html.P(
                                "Compare shot locations and outcomes across both leagues.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Interactive shot map",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Shot location, outcome, and "
                                                "attacking patterns.",
                                                className="section-subtitle",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Label(
                                                "MAP VIEW",
                                                htmlFor="shot-map-view",
                                                className="map-view-label",
                                            ),
                                            dcc.RadioItems(
                                                id="shot-map-view",
                                                options=[
                                                    {
                                                        "label": "Hybrid",
                                                        "value": "hybrid",
                                                    },
                                                    {
                                                        "label": "Individual shots",
                                                        "value": "individual",
                                                    },
                                                    {
                                                        "label": "Heatmap",
                                                        "value": "heatmap",
                                                    },
                                                ],
                                                value="hybrid",
                                                inline=True,
                                                className="shot-map-toggle",
                                                labelClassName="shot-map-option",
                                                inputClassName="shot-map-radio",
                                            ),
                                        ],
                                        className="map-view-control",
                                    ),
                                ],
                                className="shot-map-toolbar",
                            ),
                            html.P(
                                map_coverage_note,
                                id="shot-map-coverage-note",
                                className="map-coverage-note",
                            ),
                            html.P(
                                "Heat intensity represents shot frequency, not "
                                "scoring probability. Density bins are 4 × 4 "
                                "StatsBomb coordinate units.",
                                className="map-method-note",
                            ),
                            dcc.Graph(
                                id="shot-map-graph",
                                figure=create_shot_map(shots, mode="hybrid"),
                                config={
                                    "displayModeBar": False,
                                    "responsive": True,
                                },
                                className="shot-map-graph",
                            ),
                        ],
                        className="visualization-card shot-map-card",
                    ),
                ],
                className="dashboard-section",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.P(
                                "02  /  SHOT ASSOCIATIONS",
                                className="section-kicker",
                            ),
                            html.H2(
                                "How shot characteristics connect",
                                className="section-title",
                            ),
                            html.P(
                                "Categorical associations across play pattern, "
                                "shot location, body part, and outcome. Stages "
                                "show associations, not a sequence or cause. "
                                "Hover percentages are source-category shares, "
                                "not goal conversion rates. Missing outcomes "
                                "are shown as Unknown. Play patterns and body "
                                "parts below 1% are grouped as Other.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label(
                                                "SHOW",
                                                htmlFor="sankey-mode",
                                                className="map-view-label",
                                            ),
                                            dcc.RadioItems(
                                                id="sankey-mode",
                                                options=[
                                                    {
                                                        "label": "All shots",
                                                        "value": "all",
                                                    },
                                                    {
                                                        "label": "Goals only",
                                                        "value": "goals",
                                                    },
                                                ],
                                                value="all",
                                                inline=True,
                                                className="sankey-toggle",
                                                labelClassName="shot-map-option",
                                                inputClassName="shot-map-radio",
                                            ),
                                        ],
                                        className="map-view-control",
                                    ),
                                    html.Div(
                                        [
                                            html.P(
                                                f"{total_shots:,} shots represented",
                                                id="sankey-observation-count",
                                                className="sankey-observation-count",
                                                **{"aria-live": "polite"},
                                            ),
                                            html.P(
                                                "Showing all shots across both "
                                                "leagues. Link widths are rescaled "
                                                "between modes and should not be "
                                                "compared as absolute counts.",
                                                id="sankey-mode-description",
                                                className="sankey-mode-description",
                                                role="status",
                                                **{"aria-live": "polite"},
                                            ),
                                        ],
                                        className="sankey-mode-copy",
                                    ),
                                ],
                                className="sankey-toolbar",
                            ),
                            html.Div(
                                dcc.Graph(
                                    id="sankey-all-graph",
                                    figure=create_sankey_figure(shots),
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                    className="sankey-graph",
                                ),
                                id="sankey-all-view",
                            ),
                            html.Div(
                                dcc.Graph(
                                    id="sankey-goals-graph",
                                    figure=create_sankey_figure(shots, mode="goals"),
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                    className="sankey-graph",
                                ),
                                id="sankey-goals-view",
                                style={"display": "none"},
                            ),
                        ],
                        className="visualization-card sankey-card",
                    ),
                ],
                className="dashboard-section",
            ),
            html.Section(
                [
                    html.Div(
                        [
                            html.P("03  /  LEAGUE COMPARISON", className="section-kicker"),
                            html.H2("Two leagues, different profiles", className="section-title"),
                            html.P(
                                "Compare finishing efficiency and the choices "
                                "behind each attempt.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Shot conversion comparison",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Goal conversion by shot location. "
                                                "Differences are descriptive and "
                                                "do not imply statistical "
                                                "significance.",
                                                className="section-subtitle",
                                            ),
                                        ],
                                        className="visualization-heading",
                                    ),
                                    dcc.Graph(
                                        id="shot-conversion-chart",
                                        figure=conversion_figure,
                                        config={
                                            "displayModeBar": False,
                                            "responsive": True,
                                        },
                                        className="conversion-chart",
                                    ),
                                    html.P(
                                        f"Shots with missing location excluded: "
                                        f"{conversion_missing_locations:,}. "
                                        "Groups with no shots have no conversion "
                                        "rate.",
                                        className="conversion-method-note",
                                    ),
                                ],
                                className="visualization-card conversion-card",
                            ),
                            _chart_placeholder(
                                "Shot characteristics",
                                "Compare shot type, body part, and outcome.",
                                "shot-characteristics-placeholder",
                                "PHASE 04",
                            ),
                        ],
                        className="comparison-grid",
                    ),
                ],
                className="dashboard-section",
            ),
            html.Footer(
                [
                    html.Div(
                        [
                            html.P("METHODOLOGY", className="footer-label"),
                            html.P(
                                "This dashboard analyzes available StatsBomb shot-event "
                                "data for the NWSL 2023 season and FA Women's Super "
                                "League 2023/24 season. Available StatsBomb matches "
                                "may not represent every match played in either "
                                "competition.",
                                className="footer-copy",
                            ),
                        ]
                    ),
                    html.A(
                        "StatsBomb Open Data repository ↗",
                        href="https://github.com/statsbomb/open-data",
                        target="_blank",
                        rel="noreferrer",
                        className="footer-link",
                    ),
                ],
                className="dashboard-footer",
            ),
        ],
        className="dashboard-shell",
    )

    @app.callback(
        Output("shot-map-graph", "figure"),
        Input("shot-map-view", "value"),
    )
    def update_shot_map(mode: str):
        return create_shot_map(shots, mode=mode)

    @app.callback(
        Output("sankey-all-view", "style"),
        Output("sankey-goals-view", "style"),
        Output("sankey-observation-count", "children"),
        Output("sankey-mode-description", "children"),
        Input("sankey-mode", "value"),
    )
    def update_sankey(mode: str):
        if mode == "goals":
            selected_count = int(shots["shot_outcome"].eq("Goal").sum())
            description = (
                "Showing goals only. This three-stage view omits Outcome; "
                "observations are from both leagues. "
                "Link widths are rescaled between modes and should not be "
                "compared as absolute counts."
            )
            observation_label = "goals"
        elif mode == "all":
            selected_count = len(shots)
            description = (
                "Showing all shots across both leagues. Link widths are rescaled "
                "between modes and should not be compared as absolute counts."
            )
            observation_label = "shots"
        else:
            raise ValueError(f"Unsupported Sankey mode: {mode}")

        return (
            {"display": "block" if mode == "all" else "none"},
            {"display": "block" if mode == "goals" else "none"},
            f"{selected_count:,} {observation_label} represented",
            description,
        )

    return app


def main() -> None:
    app = create_app(load_local_shots())
    app.run(debug=False)


if __name__ == "__main__":
    main()
