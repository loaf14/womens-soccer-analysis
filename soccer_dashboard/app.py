"""Minimal Dash app for the prepared women's soccer shot dataset."""

import pandas as pd
from dash import Dash, html

from soccer_dashboard.storage import load_local_shots


def create_app(shots: pd.DataFrame) -> Dash:
    """Create the dashboard app using an already loaded shot dataset."""
    app = Dash(__name__, title="Women's Soccer Analytics Dashboard")
    app.layout = html.Main(
        [
            html.H1("Women's Soccer Analytics Dashboard"),
            html.Section(
                [
                    html.H2("Total shots"),
                    html.P(f"{len(shots):,}", id="total-shot-count"),
                ],
                **{"aria-label": "Unfiltered shot total"},
            ),
            html.P(
                [
                    "Data: ",
                    html.A(
                        "StatsBomb Open Data",
                        href="https://github.com/statsbomb/open-data",
                    ),
                    ".",
                ]
            ),
        ]
    )
    return app


def main() -> None:
    app = create_app(load_local_shots())
    app.run(debug=False)


if __name__ == "__main__":
    main()
