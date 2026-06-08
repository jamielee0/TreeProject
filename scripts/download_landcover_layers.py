"""Acquire NLCD impervious/landcover and USFS tree-canopy layers.

The script downloads native 30 m source layers from Google Earth Engine, then
aligns them to the fixed 70 m Phoenix reference grid. Continuous percentage
layers are resampled with averaging and saved as fractions. The NLCD land-cover
class layer is resampled with nearest neighbor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import ee
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from create_reference_grid import create_reference_grid


NLCD_COLLECTION = "USGS/NLCD_RELEASES/2021_REL/NLCD"
TCC_COLLECTION = "USGS/NLCD_RELEASES/2023_REL/TCC/v2023-5"
DEFAULT_NLCD_YEAR = 2021
DEFAULT_TCC_YEAR = 2023
SOURCE = "Google Earth Engine"
RAW_SUBDIR = "landcover"
INTERIM_SUBDIR = "landcover"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and align NLCD impervious/landcover and USFS TCC layers."
    )
    parser.add_argument("--dry-run", action="store_true", help="Search only; do not download.")
    parser.add_argument("--project", default=None, help="Optional Google Cloud project ID.")
    parser.add_argument("--nlcd-year", type=int, default=DEFAULT_NLCD_YEAR)
    parser.add_argument("--tcc-year", type=int, default=DEFAULT_TCC_YEAR)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
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
            "If Earth Engine requires a Cloud project, rerun with:\n\n"
            "  python scripts/download_landcover_layers.py --project YOUR_PROJECT_ID\n"
        ) from exc


def earth_engine_aoi() -> ee.Geometry:
    lon_min, lon_max, lat_min, lat_max = config.PHOENIX_BBOX
    return ee.Geometry.Rectangle(
        [lon_min, lat_min, lon_max, lat_max],
        proj="EPSG:4326",
        geodesic=False,
    )


def get_nlcd_image(year: int) -> ee.Image:
    collection = ee.ImageCollection(NLCD_COLLECTION).filter(ee.Filter.eq("system:index", str(year)))
    count = collection.size().getInfo()
    if count == 0:
        raise SystemExit(f"No NLCD image found for year {year} in {NLCD_COLLECTION}.")
    return ee.Image(collection.first())


def get_tcc_image(year: int) -> ee.Image:
    collection = (
        ee.ImageCollection(TCC_COLLECTION)
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .filter(ee.Filter.eq("study_area", "CONUS"))
    )
    count = collection.size().getInfo()
    if count == 0:
        raise SystemExit(f"No CONUS TCC image found for year {year} in {TCC_COLLECTION}.")
    return ee.Image(collection.first())


def output_paths(raw_dir: Path, interim_dir: Path, nlcd_year: int, tcc_year: int) -> dict[str, Path]:
    crs = config.PROJECTED_CRS.lower().replace(":", "")
    scale = f"{config.GRID_CELL_SIZE_M}m"
    return {
        "raw_impervious": raw_dir / f"nlcd_{nlcd_year}_impervious_percent_native30m.tif",
        "raw_landcover": raw_dir / f"nlcd_{nlcd_year}_landcover_native30m.tif",
        "raw_tcc": raw_dir / f"usfs_tcc_{tcc_year}_nlcd_percent_tree_canopy_native30m.tif",
        "impervious_fraction": interim_dir / f"nlcd_{nlcd_year}_impervious_fraction_{crs}_{scale}.tif",
        "tree_canopy_fraction": interim_dir / f"usfs_tcc_{tcc_year}_tree_canopy_fraction_{crs}_{scale}.tif",
        "landcover_class": interim_dir / f"nlcd_{nlcd_year}_landcover_class_{crs}_{scale}.tif",
    }


def download_ee_image(image: ee.Image, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"Source exists, skipping: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "name": path.stem,
        "region": earth_engine_aoi(),
        "scale": 30,
        "crs": config.PROJECTED_CRS,
        "format": "GEO_TIFF",
        "filePerBand": False,
    }
    url = image.clip(earth_engine_aoi()).getDownloadURL(params)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"Downloading {path.name}...")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

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


def align_to_reference(
    src_path: Path,
    dst_path: Path,
    reference_path: Path,
    resampling: Resampling,
    output_dtype: str,
    dst_nodata: float | int,
    divide_by_100: bool = False,
    overwrite: bool = False,
) -> None:
    if dst_path.exists() and not overwrite:
        print(f"Aligned output exists, skipping: {dst_path}")
        return

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(reference_path) as ref, rasterio.open(src_path) as src:
        profile = ref.profile.copy()
        profile.update(
            dtype=output_dtype,
            count=1,
            nodata=dst_nodata,
            compress="deflate",
            tiled=True,
        )
        destination = np.full((ref.height, ref.width), dst_nodata, dtype=output_dtype)
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )

    if divide_by_100:
        destination = destination.astype("float32") / 100.0
        destination = np.where(np.isfinite(destination), np.clip(destination, 0.0, 1.0), np.nan)
        profile.update(dtype="float32", nodata=np.nan)

    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(destination, 1)


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


def append_manifest_rows(paths: Iterable[Path], manifest_path: Path, notes: str) -> None:
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
                    "dataset": "NLCD impervious/landcover and USFS TCC",
                    "version": f"{NLCD_COLLECTION}; {TCC_COLLECTION}",
                    "download_date": download_date,
                    "spatial_extent": spatial_extent,
                    "time_extent": time_extent,
                    "checksum": sha256(path),
                    "notes": notes,
                }
            )


def main() -> None:
    args = parse_args()
    initialize_earth_engine(args.project)

    nlcd = get_nlcd_image(args.nlcd_year)
    tcc = get_tcc_image(args.tcc_year)
    print(f"Using NLCD {args.nlcd_year}: {NLCD_COLLECTION}")
    print(f"Using USFS TCC {args.tcc_year}: {TCC_COLLECTION}")

    if args.dry_run:
        print("Dry run only. No files downloaded.")
        return

    reference_path = create_reference_grid()
    raw_dir = config.RAW_DIR / RAW_SUBDIR
    interim_dir = config.INTERIM_DIR / INTERIM_SUBDIR
    paths = output_paths(raw_dir, interim_dir, args.nlcd_year, args.tcc_year)

    nlcd_impervious = nlcd.select("impervious").rename("impervious_percent")
    nlcd_landcover = nlcd.select("landcover").rename("landcover")
    tcc_canopy = (
        tcc.select("NLCD_Percent_Tree_Canopy_Cover")
        .rename("tree_canopy_percent")
        .updateMask(tcc.select("data_mask").eq(1))
    )

    download_ee_image(nlcd_impervious, paths["raw_impervious"], args.overwrite)
    download_ee_image(nlcd_landcover, paths["raw_landcover"], args.overwrite)
    download_ee_image(tcc_canopy, paths["raw_tcc"], args.overwrite)

    align_to_reference(
        paths["raw_impervious"],
        paths["impervious_fraction"],
        reference_path,
        Resampling.average,
        "float32",
        np.nan,
        divide_by_100=True,
        overwrite=args.overwrite,
    )
    align_to_reference(
        paths["raw_tcc"],
        paths["tree_canopy_fraction"],
        reference_path,
        Resampling.average,
        "float32",
        np.nan,
        divide_by_100=True,
        overwrite=args.overwrite,
    )
    align_to_reference(
        paths["raw_landcover"],
        paths["landcover_class"],
        reference_path,
        Resampling.nearest,
        "uint8",
        0,
        overwrite=args.overwrite,
    )

    append_manifest_rows(
        paths.values(),
        config.PROJECT_ROOT / "manifest.csv",
        notes=(
            "Section 5 land-cover acquisition; continuous percentage layers "
            "averaged to 70 m and saved as fractions; landcover nearest-neighbor"
        ),
    )
    print(f"Saved raw source layers to {raw_dir}.")
    print(f"Saved aligned 70 m layers to {interim_dir}.")
    print("Updated manifest.csv.")


if __name__ == "__main__":
    main()
