"""Local Parquet storage for generated shot data."""

from pathlib import Path

import pandas as pd

from soccer_dashboard.data import OUTPUT_COLUMNS

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "shots.parquet"


def load_local_shots(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load prepared shot data or explain how to generate it."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared shot data not found at {data_path}. "
            "Generate it with `python -m soccer_dashboard.build_data`."
        )

    shots = pd.read_parquet(data_path)
    missing_columns = set(OUTPUT_COLUMNS).difference(shots.columns)
    if missing_columns:
        raise ValueError(
            f"Prepared shot data at {data_path} is missing columns: "
            f"{sorted(missing_columns)}"
        )
    return shots.loc[:, OUTPUT_COLUMNS]
