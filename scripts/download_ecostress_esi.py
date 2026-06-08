"""Download ECOSTRESS L4T ESI evaporative-stress granules.

This script searches NASA Earthdata for ECO_L4T_ESI V002 granules within the
fixed Phoenix pilot domain and time window from config.py. Successful downloads
are saved under data/raw/ecostress_esi/ and recorded in manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import earthaccess
from earthaccess.exceptions import LoginStrategyUnavailable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config


DATASET = "ECOSTRESS Tiled Evaporative Stress Index PT-JPL Instantaneous"
SHORT_NAME = "ECO_L4T_ESI"
VERSION = "002"
SOURCE = "NASA LP DAAC Earthdata"
RAW_SUBDIR = "ecostress_esi"


def earthdata_bbox() -> tuple[float, float, float, float]:
    """Return bbox as west, south, east, north for Earthdata queries."""
    lon_min, lon_max, lat_min, lat_max = config.PHOENIX_BBOX
    return lon_min, lat_min, lon_max, lat_max


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_manifest_paths(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["file_path"] for row in csv.DictReader(handle) if row.get("file_path")}


def append_manifest_rows(paths: Iterable[Path], manifest_path: Path) -> None:
    fieldnames = [
        "file_path",
        "source",
        "dataset",
        "version",
        "download_date",
        "spatial_extent",
        "time_extent",
        "checksum",
        "notes",
    ]
    already_recorded = existing_manifest_paths(manifest_path)
    manifest_exists = manifest_path.exists()
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spatial_extent = ",".join(str(value) for value in config.PHOENIX_BBOX)
    time_extent = f"{config.PILOT_ANALYSIS_START}/{config.PILOT_ANALYSIS_END}"

    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not manifest_exists or manifest_path.stat().st_size == 0:
            writer.writeheader()

        for path in paths:
            relative_path = path.relative_to(config.PROJECT_ROOT).as_posix()
            if relative_path in already_recorded:
                continue
            writer.writerow(
                {
                    "file_path": relative_path,
                    "source": SOURCE,
                    "dataset": DATASET,
                    "version": VERSION,
                    "download_date": download_date,
                    "spatial_extent": spatial_extent,
                    "time_extent": time_extent,
                    "checksum": sha256(path),
                    "notes": SHORT_NAME,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search and download ECO_L4T_ESI V002 granules for Phoenix 2023."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Search only; print the number of matching granules without downloading.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Download only the first N matching granules. Useful for testing.",
    )
    parser.add_argument(
        "--login-strategy",
        default="interactive",
        choices=["interactive", "netrc", "environment"],
        help="Earthdata login strategy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = config.RAW_DIR / RAW_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Logging in to NASA Earthdata...")
    try:
        earthaccess.login(strategy=args.login_strategy, persist=True)
    except (EOFError, LoginStrategyUnavailable) as exc:
        raise SystemExit(
            "Earthdata login is not available in this non-interactive run.\n"
            "Run this once in your own PowerShell terminal, then rerun the script:\n\n"
            "  python -c \"import earthaccess; earthaccess.login(strategy='interactive', persist=True)\"\n"
        ) from exc

    print("Searching for ECOSTRESS ESI granules...")
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        version=VERSION,
        bounding_box=earthdata_bbox(),
        temporal=(config.PILOT_ANALYSIS_START, config.PILOT_ANALYSIS_END),
    )

    print(f"Found {len(results)} matching granules.")
    if args.dry_run:
        return
    if not results:
        print("No matching granules found. Nothing to download.")
        return

    if args.limit is not None:
        results = results[: args.limit]
        print(f"Downloading first {len(results)} granules because --limit was set.")

    downloaded = earthaccess.download(results, local_path=str(output_dir))
    downloaded_paths = [Path(path).resolve() for path in downloaded]
    append_manifest_rows(downloaded_paths, config.PROJECT_ROOT / "manifest.csv")

    print(f"Downloaded {len(downloaded_paths)} files to {output_dir}.")
    print("Updated manifest.csv.")


if __name__ == "__main__":
    main()
