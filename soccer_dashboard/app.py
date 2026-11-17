"""Dash application for women's soccer shot analysis."""

from pathlib import Path

import pandas as pd
from dash import Dash, Input, Output, dcc, html

from soccer_dashboard.sankey import create_sankey_figure, sankey_category_orders
from soccer_dashboard.shot_map import (
    create_shot_map,
    max_shot_density,
    prepare_shot_coordinates,
)
from soccer_dashboard.league_kpis import prepare_league_metrics
from soccer_dashboard.shot_conversion import (
    create_assist_conversion_figure,
    create_play_pattern_conversion_figure,
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
    league_metrics = prepare_league_metrics(shots).set_index("competition_name")
    play_pattern_conversion_figure = create_play_pattern_conversion_figure(shots)
    assist_conversion_figure = create_assist_conversion_figure(shots)
    league_shots = {
        "nwsl": shots.loc[shots["competition_name"].eq("NWSL")].copy(),
        "wsl": shots.loc[
            shots["competition_name"].eq("FA Women's Super League")
        ].copy(),
    }
    map_exclusion_counts = {}
    for league, league_data in league_shots.items():
        map_shots = prepare_shot_coordinates(league_data)
        shots_outside_view = int(
            (
                map_shots["has_valid_location"]
                & ~map_shots["in_shot_map_view"]
            ).sum()
        )
        shots_without_location = int((~map_shots["has_valid_location"]).sum())
        map_exclusion_counts[league] = {
            "outside_view": shots_outside_view,
            "unavailable_location": shots_without_location,
        }
    shared_density_max = max(
        max_shot_density(league_data) for league_data in league_shots.values()
    )
    sankey_orders = {
        mode: sankey_category_orders(league_shots, mode=mode)
        for mode in ("all", "goals")
    }

    seasons = (
        shots[["competition_name", "season_name"]]
        .drop_duplicates()
        .assign(
            _competition_order=lambda rows: rows["competition_name"].map(
                {"NWSL": 0, "FA Women's Super League": 1}
            )
        )
        .sort_values(["_competition_order", "season_name"])
        .drop(columns="_competition_order")
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
                                                        "label": "Binned shots",
                                                        "value": "binned",
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
                                className="shot-map-toolbar shot-map-toolbar-controls",
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
                                                "WSL · 2023/24",
                                                className="shot-map-league-title wsl",
                                                id="wsl-shot-map-title",
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
                                "Compare shot patterns and outcomes across leagues.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.P(
                                        "Link widths show shot counts; each league "
                                        "diagram is scaled independently. Hover "
                                        "percentages show the share of the source "
                                        "category, not goal conversion rates.",
                                        className="sankey-method-note",
                                    ),
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
                                        className="sankey-toolbar-controls",
                                    ),
                                ],
                                className="sankey-toolbar",
                            ),
                            html.Div(
                                [
                                    html.Section(
                                        [
                                            html.H3(
                                                "NWSL · 2023",
                                                className="sankey-league-title nwsl",
                                                id="nwsl-sankey-title",
                                            ),
                                            html.P(
                                                f"{len(league_shots['nwsl']):,} shots represented",
                                                id="nwsl-sankey-count",
                                                className="sankey-league-count",
                                                **{"aria-live": "polite"},
                                            ),
                                            dcc.Graph(
                                                id="nwsl-sankey-graph",
                                                figure=create_sankey_figure(
                                                    league_shots["nwsl"],
                                                    category_orders=sankey_orders["all"],
                                                ),
                                                config={
                                                    "displayModeBar": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "500px"},
                                                className="sankey-graph",
                                            ),
                                        ],
                                        className="sankey-league-panel",
                                    ),
                                    html.Section(
                                        [
                                            html.H3(
                                                "WSL · 2023/24",
                                                className="sankey-league-title wsl",
                                                id="wsl-sankey-title",
                                            ),
                                            html.P(
                                                f"{len(league_shots['wsl']):,} shots represented",
                                                id="wsl-sankey-count",
                                                className="sankey-league-count",
                                                **{"aria-live": "polite"},
                                            ),
                                            dcc.Graph(
                                                id="wsl-sankey-graph",
                                                figure=create_sankey_figure(
                                                    league_shots["wsl"],
                                                    category_orders=sankey_orders["all"],
                                                ),
                                                config={
                                                    "displayModeBar": False,
                                                    "responsive": True,
                                                },
                                                style={"height": "500px"},
                                                className="sankey-graph",
                                            ),
                                        ],
                                        className="sankey-league-panel",
                                    ),
                                ],
                                className="sankey-comparison-grid",
                                id="sankey-comparison-grid",
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
                            html.P(
                                "03  /  GOAL-SCORING EFFICIENCY",
                                className="section-kicker",
                            ),
                            html.H2(
                                "What influences shot conversion?",
                                className="section-title",
                            ),
                            html.P(
                                "Compare goal conversion across play patterns "
                                "and recorded assist status.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    html.Div(
                        [
                            html.Section(
                                [
                                    html.H3(
                                        "Conversion by play pattern",
                                        className="comparison-chart-title",
                                    ),
                                    dcc.Graph(
                                        id="play-pattern-conversion-chart",
                                        figure=play_pattern_conversion_figure,
                                        config={"displayModeBar": False},
                                        responsive=True,
                                        className="conversion-comparison-chart",
                                        style={
                                            "height": (
                                                f"{play_pattern_conversion_figure.layout.height}px"
                                            )
                                        },
                                    ),
                                    html.P(
                                        "Other play patterns excluded from this "
                                        "comparison.",
                                        className="conversion-chart-note",
                                    ),
                                ],
                                className="visualization-card conversion-card",
                            ),
                            html.Section(
                                [
                                    html.H3(
                                        "Conversion by assist status",
                                        className="comparison-chart-title",
                                    ),
                                    dcc.Graph(
                                        id="assist-conversion-chart",
                                        figure=assist_conversion_figure,
                                        config={"displayModeBar": False},
                                        responsive=True,
                                        className="conversion-comparison-chart",
                                        style={
                                            "height": (
                                                f"{assist_conversion_figure.layout.height}px"
                                            )
                                        },
                                    ),
                                ],
                                className="visualization-card conversion-card",
                            ),
                        ],
                        className="comparison-grid goal-conversion-grid",
                    ),
                    html.P(
                        "Conversion differences are descriptive, not proof of "
                        "causation or statistical significance. Unassisted means "
                        "no recorded key pass, not necessarily a chance created "
                        "entirely alone.",
                        className="conversion-method-note",
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
                        [
                            html.Img(
                                src="/assets/statsbomb-logo.svg",
                                alt="StatsBomb logo",
                                className="statsbomb-logo",
                            ),
                            html.Span("StatsBomb Open Data repository ↗"),
                        ],
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
        Output("nwsl-sankey-graph", "figure"),
        Output("wsl-sankey-graph", "figure"),
        Output("nwsl-sankey-count", "children"),
        Output("wsl-sankey-count", "children"),
        Input("sankey-mode", "value"),
    )
    def update_sankey(mode: str):
        if mode not in {"all", "goals"}:
            raise ValueError(f"Unsupported Sankey mode: {mode}")

        selected_orders = sankey_orders[mode]
        selected_league_shots = {}
        figures = {}
        for league, league_data in league_shots.items():
            selected = (
                league_data.loc[league_data["shot_outcome"].eq("Goal")]
                if mode == "goals"
                else league_data
            )
            selected_league_shots[league] = selected
            figures[league] = create_sankey_figure(
                league_data,
                mode=mode,
                category_orders=selected_orders,
            )

        observation_label = "goals" if mode == "goals" else "shots"
        return (
            figures["nwsl"],
            figures["wsl"],
            f"{len(selected_league_shots['nwsl']):,} {observation_label} represented",
            f"{len(selected_league_shots['wsl']):,} {observation_label} represented",
        )

    return app


def main() -> None:
    app = create_app(load_local_shots())
    app.run(debug=False)


def create_server():
    """Create the WSGI server for Gunicorn using the prepared local dataset."""
    return create_app(load_local_shots()).server


if __name__ == "__main__":
    main()
