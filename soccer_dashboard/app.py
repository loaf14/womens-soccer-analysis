"""Dash application for women's soccer shot analysis."""

from pathlib import Path

import pandas as pd
from dash import Dash, dcc, html

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
                                "Compare the location and outcome of every attempt "
                                "across both leagues.",
                                className="section-subtitle",
                            ),
                        ],
                        className="section-heading",
                    ),
                    _chart_placeholder(
                        "Interactive shot map",
                        "Location, outcome, and attacking patterns",
                        "shot-map-placeholder",
                        "PHASE 03",
                        "shot-map-card",
                    ),
                ],
                className="dashboard-section",
            ),
            html.Section(
                [
                    html.P(
                        "02  /  ATTACK SEQUENCE",
                        className="section-kicker",
                    ),
                    _chart_placeholder(
                        "How shot characteristics connect",
                        "Explore associations among play pattern, shot location, "
                        "body part, and outcome.",
                        "sankey-placeholder",
                        "PHASE 04",
                        "sankey-card",
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
                            _chart_placeholder(
                                "Shot conversion comparison",
                                "Goals scored relative to total attempts.",
                                "shot-conversion-placeholder",
                                "PHASE 04",
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
    return app


def main() -> None:
    app = create_app(load_local_shots())
    app.run(debug=False)


if __name__ == "__main__":
    main()
