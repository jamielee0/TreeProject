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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config


DATASET = "ECOSTRESS Tiled Evaporative Stress Index PT-JPL Instantaneous"
SHORT_NAME = "ECO_L4T_ESI"
VERSION = "002"
SOURCE = "NASA LP DAAC Earthdata"
RAW_SUBDIR = "ecostress_esi"
MANIFEST_FIELDNAMES = [
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


def valid_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    rows: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            file_path = row.get("file_path", "")
            if not file_path.startswith("data/"):
                continue
            rows.append({field: row.get(field, "") for field in MANIFEST_FIELDNAMES})
    return rows


def manifest_row(path: Path, download_date: str) -> dict[str, str]:
    relative_path = path.relative_to(config.PROJECT_ROOT).as_posix()
    spatial_extent = ",".join(str(value) for value in config.PHOENIX_BBOX)
    time_extent = f"{config.PILOT_ANALYSIS_START}/{config.PILOT_ANALYSIS_END}"
    return {
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


def sync_manifest_rows(paths: Iterable[Path], manifest_path: Path) -> int:
    rows_added = 0
    rows = valid_manifest_rows(manifest_path)
    seen = {row["file_path"] for row in rows}
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for path in paths:
        relative_path = path.relative_to(config.PROJECT_ROOT).as_posix()
        if relative_path in seen:
            continue
        rows.append(manifest_row(path, download_date))
        seen.add(relative_path)
        rows_added += 1

    temp_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(manifest_path)

    return rows_added


def local_product_paths(output_dir: Path) -> list[Path]:
    return sorted(path.resolve() for path in output_dir.glob("*.tif"))


def import_earthaccess():
    try:
        import earthaccess
        from earthaccess.exceptions import LoginStrategyUnavailable
    except ImportError as exc:
        raise SystemExit(
            "earthaccess is required for Earthdata search/download. "
            "Use --sync-manifest-only to record already-downloaded local files."
        ) from exc
    return earthaccess, LoginStrategyUnavailable


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
    parser.add_argument(
        "--sync-manifest-only",
        action="store_true",
        help="Do not contact Earthdata; record existing local product TIFFs in manifest.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = config.RAW_DIR / RAW_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.PROJECT_ROOT / "manifest.csv"

    if args.sync_manifest_only:
        local_paths = local_product_paths(output_dir)
        rows_added = sync_manifest_rows(local_paths, manifest_path)
        print(f"Found {len(local_paths)} local ECOSTRESS ESI files in {output_dir}.")
        print(f"Added {rows_added} manifest rows.")
        return

    earthaccess, LoginStrategyUnavailable = import_earthaccess()

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
    local_paths = local_product_paths(output_dir)
    rows_added = sync_manifest_rows(local_paths, manifest_path)

    print(f"Downloaded {len(downloaded_paths)} files to {output_dir}.")
    print(f"Synced manifest with {len(local_paths)} local files; added {rows_added} rows.")


if __name__ == "__main__":
    main()
