"""Download StatsBomb seasons and cache their shots in a local Parquet file."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from soccer_dashboard.data import load_shots
from soccer_dashboard.storage import DEFAULT_DATA_PATH

SEASONS = ((49, 107), (37, 281))


def build_shots(
    output_path: str | Path = DEFAULT_DATA_PATH,
    expected_total: int | None = None,
) -> pd.DataFrame:
    """Fetch both configured seasons and save their combined shots locally."""
    seasons = [load_shots(competition_id, season_id) for competition_id, season_id in SEASONS]
    shots = pd.concat(seasons, ignore_index=True)

    if expected_total is not None and len(shots) != expected_total:
        raise ValueError(
            f"Expected {expected_total:,} shots across configured seasons; "
            f"StatsBomb currently returns {len(shots):,}."
        )

    data_path = Path(output_path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = data_path.with_name(f"{data_path.name}.tmp")
    try:
        shots.to_parquet(temporary_path, index=False)
        temporary_path.replace(data_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help=f"Parquet output path (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        help="Fail if the combined current source count differs from this value.",
    )
    args = parser.parse_args()

    shots = build_shots(args.output, expected_total=args.expected_total)
    print(f"Saved {len(shots):,} shots to {args.output}")
    print(
        shots.groupby(["competition_name", "season_name"], dropna=False)
        .size()
        .to_string()
    )


if __name__ == "__main__":
    main()
