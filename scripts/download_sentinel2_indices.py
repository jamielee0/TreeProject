"""Create Sentinel-2 NDVI and NDMI warm-season composites with Earth Engine.

This script uses COPERNICUS/S2_SR_HARMONIZED for the Phoenix pilot window,
masks cloud/cloud-shadow/snow pixels with SCL, computes per-scene NDVI and
NDMI, reduces each index to a warm-season median composite, and downloads the
outputs as GeoTIFFs in EPSG:32612 at 70 m.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import math
import sys
import time
import urllib.request
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import ee
import numpy as np
import rasterio
from rasterio.windows import Window

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from create_reference_grid import create_reference_grid


COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
DATASET = "Sentinel-2 Surface Reflectance Harmonized NDVI/NDMI composites"
VERSION = COLLECTION
SOURCE = "Google Earth Engine"
OUTPUT_SUBDIR = "sentinel2"
SCL_MASK_VALUES = (0, 1, 3, 8, 9, 10, 11)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and download Sentinel-2 NDVI/NDMI median composites."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate, search the collection, and print scene count only.",
    )
    parser.add_argument(
        "--max-cloud-percent",
        type=float,
        default=30.0,
        help="Maximum CLOUDY_PIXEL_PERCENTAGE allowed for input scenes.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Optional Google Cloud project ID for Earth Engine initialization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing GeoTIFF outputs.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Number of download attempts per output file.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Download timeout in seconds per request.",
    )
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=4,
        help="Number of tile rows for Earth Engine downloads.",
    )
    parser.add_argument(
        "--tile-cols",
        type=int,
        default=4,
        help="Number of tile columns for Earth Engine downloads.",
    )
    parser.add_argument(
        "--keep-tiles",
        action="store_true",
        help="Keep temporary downloaded tile GeoTIFFs.",
    )
    return parser.parse_args()


def initialize_earth_engine(project: str | None) -> None:
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        raise SystemExit(
            "Google Earth Engine is not authenticated or initialized.\n"
            f"Earth Engine error: {detail}\n\n"
            "Run this once in your own PowerShell terminal, then rerun the script:\n\n"
            "  earthengine authenticate\n\n"
            "If Earth Engine asks for a Cloud project, rerun this script with:\n\n"
            "  python scripts/download_sentinel2_indices.py --project YOUR_PROJECT_ID\n"
        ) from exc


def pilot_end_exclusive() -> str:
    end_date = date.fromisoformat(config.PILOT_ANALYSIS_END)
    return (end_date + timedelta(days=1)).isoformat()


def earth_engine_aoi() -> ee.Geometry:
    lon_min, lon_max, lat_min, lat_max = config.PHOENIX_BBOX
    return ee.Geometry.Rectangle(
        [lon_min, lat_min, lon_max, lat_max],
        proj="EPSG:4326",
        geodesic=False,
    )


def mask_sentinel2_scene(image: ee.Image) -> ee.Image:
    scl = image.select("SCL")
    mask = ee.Image(1)
    for value in SCL_MASK_VALUES:
        mask = mask.And(scl.neq(value))
    return image.updateMask(mask)


def add_indices(image: ee.Image) -> ee.Image:
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndmi = image.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return image.addBands(ee.Image.cat([ndvi, ndmi]))


def build_collection(max_cloud_percent: float) -> ee.ImageCollection:
    return (
        ee.ImageCollection(COLLECTION)
        .filterBounds(earth_engine_aoi())
        .filterDate(config.PILOT_ANALYSIS_START, pilot_end_exclusive())
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud_percent))
        .map(mask_sentinel2_scene)
        .map(add_indices)
    )


def build_composites(collection: ee.ImageCollection) -> dict[str, ee.Image]:
    aoi = earth_engine_aoi()
    native_projection = ee.Image(collection.first()).select("B8").projection()

    def median_index(band_name: str) -> ee.Image:
        return (
            collection.select(band_name)
            .median()
            .rename(band_name)
            .clip(aoi)
            .setDefaultProjection(native_projection)
            .resample("bilinear")
        )

    return {
        "ndvi": median_index("NDVI"),
        "ndmi": median_index("NDMI"),
    }


def output_paths(output_dir: Path) -> dict[str, Path]:
    start = config.PILOT_ANALYSIS_START.replace("-", "")
    end = config.PILOT_ANALYSIS_END.replace("-", "")
    crs = config.PROJECTED_CRS.lower().replace(":", "")
    scale = f"{config.GRID_CELL_SIZE_M}m"
    prefix = f"sentinel2_phoenix_{start}_{end}"
    return {
        "ndvi": output_dir / f"{prefix}_ndvi_median_{crs}_{scale}.tif",
        "ndmi": output_dir / f"{prefix}_ndmi_median_{crs}_{scale}.tif",
    }


def finalize_download(tmp_path: Path, path: Path) -> None:
    if zipfile.is_zipfile(tmp_path):
        with zipfile.ZipFile(tmp_path) as zf:
            tif_names = [name for name in zf.namelist() if name.lower().endswith(".tif")]
            if len(tif_names) != 1:
                raise RuntimeError(f"Expected one GeoTIFF in {tmp_path}, found {tif_names}.")
            with zf.open(tif_names[0]) as source, path.open("wb") as target:
                target.write(source.read())
        tmp_path.unlink()
    else:
        tmp_path.replace(path)


def download_tile(
    image: ee.Image,
    path: Path,
    region: ee.Geometry,
    dimensions: str,
    retries: int,
    timeout: int,
) -> None:
    params = {
        "name": path.stem,
        "region": region,
        "crs": config.PROJECTED_CRS,
        "dimensions": dimensions,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = image.getDownloadURL(params)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "urban-tree-thresholds/1.0"})

    for attempt in range(1, retries + 1):
        try:
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"Downloading {path.name} (attempt {attempt}/{retries})...")
            with urllib.request.urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            finalize_download(tmp_path, path)
            return
        except (http.client.RemoteDisconnected, TimeoutError, OSError, urllib.error.URLError) as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise RuntimeError(
                    f"Failed to download {path.name} after {retries} attempts."
                ) from exc
            wait_seconds = min(60, 5 * attempt)
            print(f"Download interrupted: {exc}. Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)


def tile_windows(width: int, height: int, tile_cols: int, tile_rows: int) -> Iterable[Window]:
    tile_width = math.ceil(width / tile_cols)
    tile_height = math.ceil(height / tile_rows)
    for row_off in range(0, height, tile_height):
        for col_off in range(0, width, tile_width):
            yield Window(
                col_off=col_off,
                row_off=row_off,
                width=min(tile_width, width - col_off),
                height=min(tile_height, height - row_off),
            )


def window_region(transform: rasterio.Affine, window: Window) -> ee.Geometry:
    left = transform.c + window.col_off * transform.a
    top = transform.f + window.row_off * transform.e
    right = left + window.width * transform.a
    bottom = top + window.height * transform.e
    return ee.Geometry.Rectangle(
        [left, bottom, right, top],
        proj=config.PROJECTED_CRS,
        geodesic=False,
    )


def download_image_tiled(
    image: ee.Image,
    path: Path,
    reference_path: Path,
    overwrite: bool,
    retries: int,
    timeout: int,
    tile_rows: int,
    tile_cols: int,
    keep_tiles: bool,
) -> None:
    if path.exists() and not overwrite:
        print(f"Output exists, skipping: {path}")
        return

    tile_dir = path.parent / "_tiles" / path.stem
    tile_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()
        profile.update(
            dtype="float32",
            count=1,
            nodata=np.nan,
            compress="deflate",
            tiled=True,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(path, "w", **profile) as dst:
            for index, window in enumerate(
                tile_windows(ref.width, ref.height, tile_cols, tile_rows), start=1
            ):
                tile_width = int(window.width)
                tile_height = int(window.height)
                tile_path = tile_dir / f"{path.stem}_tile_{index:03d}.tif"
                region = window_region(ref.transform, window)
                dimensions = f"{tile_width}x{tile_height}"
                download_tile(image, tile_path, region, dimensions, retries, timeout)
                with rasterio.open(tile_path) as tile:
                    data = tile.read(
                        1,
                        out_shape=(tile_height, tile_width),
                        masked=True,
                    ).astype("float32")
                    filled = data.filled(np.nan)
                dst.write(filled, 1, window=window)
                if not keep_tiles:
                    tile_path.unlink(missing_ok=True)

    if not keep_tiles:
        try:
            tile_dir.rmdir()
            tile_dir.parent.rmdir()
        except OSError:
            pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_manifest_rows(
    paths: Iterable[Path], manifest_path: Path, max_cloud_percent: float
) -> None:
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
    download_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spatial_extent = ",".join(str(value) for value in config.PHOENIX_BBOX)
    time_extent = f"{config.PILOT_ANALYSIS_START}/{config.PILOT_ANALYSIS_END}"
    notes = (
        f"median composite; max_cloud_percent={max_cloud_percent}; "
        f"SCL masked={SCL_MASK_VALUES}; crs={config.PROJECTED_CRS}; "
        f"scale={config.GRID_CELL_SIZE_M}m; resampling=bilinear after setting native Sentinel-2 projection"
    )

    new_rows = []
    output_paths_relative = set()
    for path in paths:
        relative_path = path.relative_to(config.PROJECT_ROOT).as_posix()
        output_paths_relative.add(relative_path)
        new_rows.append(
            {
                "file_path": relative_path,
                "source": SOURCE,
                "dataset": DATASET,
                "version": VERSION,
                "download_date": download_date,
                "spatial_extent": spatial_extent,
                "time_extent": time_extent,
                "checksum": sha256(path),
                "notes": notes,
            }
        )

    retained_rows = [
        row for row in existing_manifest_rows(manifest_path)
        if row.get("file_path") not in output_paths_relative
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(retained_rows + new_rows)


def main() -> None:
    args = parse_args()
    initialize_earth_engine(args.project)

    collection = build_collection(args.max_cloud_percent)
    scene_count = collection.size().getInfo()
    print(f"Found {scene_count} Sentinel-2 scenes after filters.")
    if scene_count == 0:
        print("No matching Sentinel-2 scenes found. Nothing to download.")
        return
    if args.dry_run:
        return

    output_dir = config.INTERIM_DIR / OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = create_reference_grid()

    composites = build_composites(collection)
    paths = output_paths(output_dir)
    for key, image in composites.items():
        download_image_tiled(
            image,
            paths[key],
            reference_path,
            args.overwrite,
            args.retries,
            args.timeout,
            args.tile_rows,
            args.tile_cols,
            args.keep_tiles,
        )

    append_manifest_rows(paths.values(), config.PROJECT_ROOT / "manifest.csv", args.max_cloud_percent)
    print(f"Saved Sentinel-2 composites to {output_dir}.")
    print("Updated manifest.csv.")


if __name__ == "__main__":
    main()
