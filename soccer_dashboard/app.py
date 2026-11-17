"""Dash application for women's soccer shot analysis."""

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from soccer_dashboard.sankey import create_sankey_figure
from soccer_dashboard.shot_map import (
    create_shot_map,
    max_shot_density,
    prepare_shot_coordinates,
)
from soccer_dashboard.league_kpis import prepare_league_metrics
from soccer_dashboard.shot_conversion import (
    create_conversion_figure,
    prepare_conversion_data,
)
from soccer_dashboard.play_pattern import (
    create_play_pattern_figure,
    prepare_play_pattern_data,
)
from soccer_dashboard.storage import load_local_shots

COMPETITION_VALUES = {
    "NWSL": "nwsl",
    "FA Women's Super League": "wsl",
}


def _metric_card(label: str, value: str, card_id: str) -> html.Article:
    return html.Article(
        [
            html.P(label, className="metric-label"),
            html.P(value, id=card_id, className="metric-value"),
        ],
        className="metric-card",
    )


def _league_kpi_group(
    league: str, competition_class: str, metrics: pd.Series
) -> html.Section:
    return html.Section(
        [
            html.H2(league, className="league-kpi-title"),
            html.Div(
                [
                    _metric_card(
                        "TOTAL SHOTS",
                        f"{int(metrics['shots']):,}",
                        f"{competition_class}-shots",
                    ),
                    _metric_card(
                        "TOTAL GOALS",
                        f"{int(metrics['goals']):,}",
                        f"{competition_class}-goals",
                    ),
                    _metric_card(
                        "GOAL CONVERSION",
                        f"{metrics['conversion_rate']:.1f}%",
                        f"{competition_class}-conversion-rate",
                    ),
                    _metric_card(
                        "MATCHES ANALYZED",
                        f"{int(metrics['matches']):,}",
                        f"{competition_class}-matches",
                    ),
                    _metric_card(
                        "SHOTS PER MATCH",
                        f"{metrics['shots_per_match']:.1f}",
                        f"{competition_class}-shots-per-match",
                    ),
                ],
                className="league-kpi-metrics",
            ),
        ],
        className=f"league-kpi-group {competition_class}",
        id=f"{competition_class}-kpi-group",
    )


def create_app(shots: pd.DataFrame) -> Dash:
    """Create the dashboard using an already loaded shot dataset."""
    total_shots = len(shots)
    league_metrics = prepare_league_metrics(shots).set_index("competition_name")
    conversion_figure = create_conversion_figure(shots)
    _, conversion_missing_locations = prepare_conversion_data(shots)
    play_pattern_figure = create_play_pattern_figure(shots)
    _, missing_play_patterns = prepare_play_pattern_data(shots)
    league_shots = {
        "nwsl": shots.loc[shots["competition_name"].eq("NWSL")].copy(),
        "wsl": shots.loc[
            shots["competition_name"].eq("FA Women's Super League")
        ].copy(),
    }
    map_coverage_notes = {}
    for league, league_data in league_shots.items():
        map_shots = prepare_shot_coordinates(league_data)
        shots_outside_view = int(
            (
                map_shots["has_valid_location"]
                & ~map_shots["in_shot_map_view"]
            ).sum()
        )
        shots_without_location = int((~map_shots["has_valid_location"]).sum())
        map_coverage_notes[league] = (
            "Focused view: StatsBomb x=95–120 and y=10–70; goal margin "
            "extends beyond x=120. "
            f"Valid shots outside this window: {shots_outside_view:,}; not shown. "
            f"Shot locations unavailable or invalid: {shots_without_location:,}."
        )
    shared_density_max = max(
        max_shot_density(league_data) for league_data in league_shots.values()
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
                            _league_kpi_group(
                                "NWSL",
                                "nwsl",
                                league_metrics.loc["NWSL"],
                            ),
                            _league_kpi_group(
                                "WSL",
                                "wsl",
                                league_metrics.loc["FA Women's Super League"],
                            ),
                        ],
                        className="league-kpi-grid",
                    )
                ],
                className="dashboard-section league-kpi-section",
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
                                                "Compare shot locations",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "NWSL and WSL shots share a zoomed "
                                                "penalty-area view.",
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
                                "Heat intensity represents shot frequency, not "
                                "scoring probability. Both maps share one color "
                                "scale, so equal colors indicate the same shot "
                                "count per bin. Density bins are 4 × 4 StatsBomb "
                                "coordinate units.",
                                className="map-method-note",
                            ),
                            html.Div(
                                [
                                    html.Section(
                                        [
                                            html.H3(
                                                "NWSL · 2023",
                                                className="shot-map-league-title nwsl",
                                                id="nwsl-shot-map-title",
                                            ),
                                            html.P(
                                                map_coverage_notes["nwsl"],
                                                id="nwsl-map-coverage-note",
                                                className="map-coverage-note",
                                            ),
                                            dcc.Graph(
                                                id="nwsl-shot-map-graph",
                                                figure=create_shot_map(
                                                    league_shots["nwsl"],
                                                    mode="hybrid",
                                                    density_max=shared_density_max,
                                                ),
                                                config={
                                                    "displayModeBar": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "360px"},
                                                className="shot-map-graph",
                                            ),
                                        ],
                                        className="shot-map-panel",
                                    ),
                                    html.Section(
                                        [
                                            html.H3(
                                                "FA Women's Super League · 2023/24",
                                                className="shot-map-league-title wsl",
                                                id="wsl-shot-map-title",
                                            ),
                                            html.P(
                                                map_coverage_notes["wsl"],
                                                id="wsl-map-coverage-note",
                                                className="map-coverage-note",
                                            ),
                                            dcc.Graph(
                                                id="wsl-shot-map-graph",
                                                figure=create_shot_map(
                                                    league_shots["wsl"],
                                                    mode="hybrid",
                                                    density_max=shared_density_max,
                                                ),
                                                config={
                                                    "displayModeBar": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "360px"},
                                                className="shot-map-graph",
                                            ),
                                        ],
                                        className="shot-map-panel",
                                    ),
                                ],
                                className="shot-map-grid",
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
                                        config={"displayModeBar": False},
                                        responsive=True,
                                        className="conversion-chart",
                                        style={"height": "260px"},
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
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Where Do Shots Come From?",
                                                className="section-title",
                                            ),
                                            html.P(
                                                "Distribution of shots by play "
                                                "pattern across the NWSL and WSL.",
                                                className="section-subtitle",
                                            ),
                                        ],
                                        className="visualization-heading",
                                    ),
                                    dcc.Graph(
                                        id="play-pattern-chart",
                                        figure=play_pattern_figure,
                                        config={"displayModeBar": False},
                                        responsive=True,
                                        className="play-pattern-chart",
                                        style={
                                            "height": (
                                                f"{play_pattern_figure.layout.height}px"
                                            )
                                        },
                                    ),
                                    html.P(
                                        f"Percentages use each league's total "
                                        f"shots; missing play patterns "
                                        f"({missing_play_patterns:,}) are shown "
                                        "as a separate category. These values "
                                        "describe shot distribution, not goal "
                                        "conversion rates or the probability "
                                        "of producing a shot from an attacking "
                                        "opportunity.",
                                        className="play-pattern-method-note",
                                    ),
                                ],
                                className=(
                                    "visualization-card play-pattern-card"
                                ),
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
        Output("nwsl-shot-map-graph", "figure"),
        Output("wsl-shot-map-graph", "figure"),
        Input("shot-map-view", "value"),
    )
    def update_shot_map(mode: str):
        return (
            create_shot_map(
                league_shots["nwsl"], mode=mode, density_max=shared_density_max
            ),
            create_shot_map(
                league_shots["wsl"], mode=mode, density_max=shared_density_max
            ),
        )

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
