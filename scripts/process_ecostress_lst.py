"""Process raw ECOSTRESS L2T LSTE files into a QC-filtered LST cube.

Section 2 deliverables produced by this script:
- quality-controlled time-indexed LST cube
- usable-observation count raster
- mean LST raster for visual QA only
- simple PNG quicklook of mean LST
- per-granule QC summary CSV

The cube preserves overpass timestamps. The mean raster is not an analysis
surface; it is only a visual confirmation artifact.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import netCDF4
import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from create_reference_grid import create_reference_grid


RAW_DIR = config.RAW_DIR / "ecostress_lst"
OUTPUT_DIR = config.INTERIM_DIR / "ecostress_lst"
SUMMARY_PATH = OUTPUT_DIR / "ecostress_lst_qc_summary.csv"
CUBE_PATH = OUTPUT_DIR / "ecostress_lst_phoenix_20230601_20230930_qc_epsg32612_70m.nc"
COUNT_PATH = OUTPUT_DIR / "ecostress_lst_usable_observation_count_epsg32612_70m.tif"
MEAN_PATH = OUTPUT_DIR / "ecostress_lst_mean_visual_check_epsg32612_70m.tif"
QUICKLOOK_PATH = config.FIGURES_DIR / "ecostress_lst_mean_visual_check.png"

REQUIRED_SUFFIXES = ("LST", "QC", "cloud", "water", "height")
TIMESTAMP_RE = re.compile(r"_(\d{8}T\d{6})_")
TILE_RE = re.compile(r"_(\d{2}[A-Z]{3})_")


@dataclass(frozen=True)
class Granule:
    base: str
    timestamp: str
    tile: str
    paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC-filter and stack ECOSTRESS L2T LSTE raw GeoTIFFs."
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Index raw files and print what would be processed without writing outputs.",
    )
    parser.add_argument(
        "--limit-times",
        type=int,
        default=None,
        help="Process only the first N timestamps. Useful for testing.",
    )
    parser.add_argument("--lst-min", type=float, default=290.0, help="Minimum plausible LST in Kelvin.")
    parser.add_argument("--lst-max", type=float, default=340.0, help="Maximum plausible LST in Kelvin.")
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.001,
        help="Discard a raw granule if less than this fraction of pixels survive QC.",
    )
    parser.add_argument(
        "--allow-degraded-qc",
        action="store_true",
        help="Also keep mandatory-QA value 01. Default keeps only value 00.",
    )
    return parser.parse_args()


def output_paths(limit_times: int | None) -> dict[str, Path]:
    if limit_times is None:
        suffix = ""
    else:
        suffix = f"_first{limit_times}timestamps"
    return {
        "summary": OUTPUT_DIR / f"ecostress_lst_qc_summary{suffix}.csv",
        "cube": OUTPUT_DIR / f"ecostress_lst_phoenix_20230601_20230930_qc_epsg32612_70m{suffix}.nc",
        "count": OUTPUT_DIR / f"ecostress_lst_usable_observation_count_epsg32612_70m{suffix}.tif",
        "mean": OUTPUT_DIR / f"ecostress_lst_mean_visual_check_epsg32612_70m{suffix}.tif",
        "quicklook": config.FIGURES_DIR / f"ecostress_lst_mean_visual_check{suffix}.png",
    }


def base_without_suffix(path: Path, suffix: str) -> str:
    marker = f"_{suffix}"
    if not path.stem.endswith(marker):
        raise ValueError(f"{path.name} does not end with {marker}.")
    return path.stem[: -len(marker)]


def parse_timestamp(base: str) -> str:
    match = TIMESTAMP_RE.search(base)
    if not match:
        raise ValueError(f"Could not parse timestamp from {base}.")
    return match.group(1)


def parse_tile(base: str) -> str:
    match = TILE_RE.search(base)
    return match.group(1) if match else "unknown"


def index_granules(raw_dir: Path) -> tuple[list[Granule], list[dict[str, str]]]:
    lst_paths = sorted(raw_dir.glob("*_LST.tif"))
    granules: list[Granule] = []
    missing_rows: list[dict[str, str]] = []

    for lst_path in lst_paths:
        base = base_without_suffix(lst_path, "LST")
        paths = {suffix: raw_dir / f"{base}_{suffix}.tif" for suffix in REQUIRED_SUFFIXES}
        missing = [suffix for suffix, path in paths.items() if not path.exists()]
        if missing:
            missing_rows.append(
                {
                    "base": base,
                    "timestamp": parse_timestamp(base),
                    "tile": parse_tile(base),
                    "status": "missing_required_files",
                    "reason": ",".join(missing),
                }
            )
            continue
        granules.append(
            Granule(
                base=base,
                timestamp=parse_timestamp(base),
                tile=parse_tile(base),
                paths=paths,
            )
        )

    return granules, missing_rows


def grouped_by_timestamp(granules: Iterable[Granule]) -> dict[str, list[Granule]]:
    groups: dict[str, list[Granule]] = {}
    for granule in granules:
        groups.setdefault(granule.timestamp, []).append(granule)
    return groups


def qc_mask(
    lst: np.ndarray,
    qc: np.ndarray,
    cloud: np.ndarray,
    water: np.ndarray,
    lst_min: float,
    lst_max: float,
    allow_degraded_qc: bool,
) -> np.ndarray:
    mandatory_qa = qc & 0b11
    data_quality = (qc >> 2) & 0b11

    if allow_degraded_qc:
        mandatory_ok = (mandatory_qa == 0) | (mandatory_qa == 1)
    else:
        mandatory_ok = mandatory_qa == 0

    return (
        np.isfinite(lst)
        & mandatory_ok
        & (data_quality == 0)
        & (cloud == 0)
        & (water == 0)
        & (lst >= lst_min)
        & (lst <= lst_max)
    )


def read_clean_granule(
    granule: Granule,
    lst_min: float,
    lst_max: float,
    min_valid_fraction: float,
    allow_degraded_qc: bool,
) -> tuple[np.ndarray | None, rasterio.Affine | None, rasterio.crs.CRS | None, dict[str, str]]:
    with rasterio.open(granule.paths["LST"]) as lst_ds, rasterio.open(
        granule.paths["QC"]
    ) as qc_ds, rasterio.open(granule.paths["cloud"]) as cloud_ds, rasterio.open(
        granule.paths["water"]
    ) as water_ds:
        lst = lst_ds.read(1).astype("float32")
        qc = qc_ds.read(1)
        cloud = cloud_ds.read(1)
        water = water_ds.read(1)
        mask = qc_mask(lst, qc, cloud, water, lst_min, lst_max, allow_degraded_qc)
        valid_fraction = float(mask.sum() / mask.size)
        raw_min = float(np.nanmin(lst)) if np.isfinite(lst).any() else np.nan
        raw_max = float(np.nanmax(lst)) if np.isfinite(lst).any() else np.nan

        row = {
            "base": granule.base,
            "timestamp": granule.timestamp,
            "tile": granule.tile,
            "status": "used",
            "reason": "",
            "source_pixels": str(mask.size),
            "valid_pixels_after_qc": str(int(mask.sum())),
            "valid_fraction_after_qc": f"{valid_fraction:.8f}",
            "raw_lst_min_k": f"{raw_min:.3f}",
            "raw_lst_max_k": f"{raw_max:.3f}",
        }

        if valid_fraction < min_valid_fraction:
            row["status"] = "discarded"
            row["reason"] = "below_min_valid_fraction"
            return None, None, None, row

        clean = np.where(mask, lst, np.nan).astype("float32")
        return clean, lst_ds.transform, lst_ds.crs, row


def mosaic_timestamp(
    granules: list[Granule],
    reference: rasterio.DatasetReader,
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, list[dict[str, str]]]:
    sum_grid = np.zeros((reference.height, reference.width), dtype="float64")
    count_grid = np.zeros((reference.height, reference.width), dtype="uint16")
    rows: list[dict[str, str]] = []

    for granule in granules:
        clean, src_transform, src_crs, row = read_clean_granule(
            granule,
            args.lst_min,
            args.lst_max,
            args.min_valid_fraction,
            args.allow_degraded_qc,
        )
        rows.append(row)
        if clean is None or src_transform is None or src_crs is None:
            continue

        dest = np.full((reference.height, reference.width), np.nan, dtype="float32")
        reproject(
            source=clean,
            destination=dest,
            src_transform=src_transform,
            src_crs=src_crs,
            src_nodata=np.nan,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        valid = np.isfinite(dest)
        sum_grid[valid] += dest[valid]
        count_grid[valid] += 1

    timestamp_layer = np.full((reference.height, reference.width), np.nan, dtype="float32")
    valid_timestamp = count_grid > 0
    if not np.any(valid_timestamp):
        return None, rows
    timestamp_layer[valid_timestamp] = (
        sum_grid[valid_timestamp] / count_grid[valid_timestamp]
    ).astype("float32")
    return timestamp_layer, rows


def create_cube(path: Path, reference: rasterio.DatasetReader, timestamps: list[str]) -> netCDF4.Dataset:
    path.parent.mkdir(parents=True, exist_ok=True)
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", len(timestamps))
    ds.createDimension("y", reference.height)
    ds.createDimension("x", reference.width)

    x_coords = reference.transform.c + (np.arange(reference.width) + 0.5) * reference.transform.a
    y_coords = reference.transform.f + (np.arange(reference.height) + 0.5) * reference.transform.e
    times = np.array(
        [
            datetime.strptime(ts, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp()
            for ts in timestamps
        ],
        dtype="float64",
    )

    time_var = ds.createVariable("time", "f8", ("time",))
    y_var = ds.createVariable("y", "f8", ("y",))
    x_var = ds.createVariable("x", "f8", ("x",))
    lst_var = ds.createVariable(
        "lst",
        "f4",
        ("time", "y", "x"),
        zlib=True,
        complevel=4,
        chunksizes=(1, min(256, reference.height), min(256, reference.width)),
        fill_value=np.nan,
    )

    time_var[:] = times
    time_var.units = "seconds since 1970-01-01 00:00:00 UTC"
    time_var.calendar = "standard"
    time_var.long_name = "ECOSTRESS overpass timestamp"
    y_var[:] = y_coords
    y_var.units = "m"
    y_var.standard_name = "projection_y_coordinate"
    x_var[:] = x_coords
    x_var.units = "m"
    x_var.standard_name = "projection_x_coordinate"
    lst_var.units = "K"
    lst_var.long_name = "Quality-controlled ECOSTRESS land surface temperature"

    ds.title = "ECOSTRESS L2T LSTE quality-controlled Phoenix LST cube"
    ds.crs = str(reference.crs)
    ds.grid_cell_size_m = str(config.GRID_CELL_SIZE_M)
    ds.transform = ",".join(str(v) for v in reference.transform.to_gdal())
    ds.history = f"Created {datetime.now(timezone.utc).isoformat()}"
    ds.note = (
        "QC mask keeps mandatory QA bits 1&0 == 00, data-quality bits 3&2 == 00, "
        "cloud == 0, water == 0, and plausible LST range. Mean products are QA only."
    )
    return ds


def write_raster(path: Path, reference: rasterio.DatasetReader, data: np.ndarray, dtype: str, nodata) -> None:
    profile = reference.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def lst_to_rgb(lst: np.ndarray) -> np.ndarray:
    finite = np.isfinite(lst)
    if not finite.any():
        return np.zeros((*lst.shape, 3), dtype=np.uint8)
    lo, hi = np.nanpercentile(lst[finite], [2, 98])
    scaled = np.clip((lst - lo) / (hi - lo), 0.0, 1.0)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    cool = np.array([30, 80, 170], dtype=np.float32)
    mid = np.array([245, 240, 160], dtype=np.float32)
    hot = np.array([170, 40, 30], dtype=np.float32)
    rgb = np.empty((*lst.shape, 3), dtype=np.float32)
    low = scaled <= 0.5
    high = ~low
    rgb[low] = cool + (mid - cool) * (scaled[low, None] / 0.5)
    rgb[high] = mid + (hot - mid) * ((scaled[high, None] - 0.5) / 0.5)
    rgb[~finite] = np.array([255, 255, 255], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def write_quicklook(path: Path, mean_lst: np.ndarray) -> None:
    step = max(1, int(np.ceil(max(mean_lst.shape) / 1400)))
    preview = mean_lst[::step, ::step]
    image = Image.fromarray(lst_to_rgb(preview), mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 540, 34), fill=(255, 255, 255))
    draw.text((14, 13), "Mean ECOSTRESS LST for visual QA only - do not use as analysis average", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    paths = output_paths(args.limit_times)
    if paths["cube"].exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists: {paths['cube']}. Use --overwrite to recreate it.")

    granules, missing_rows = index_granules(RAW_DIR)
    groups = grouped_by_timestamp(granules)
    timestamps = sorted(groups)
    if args.limit_times is not None:
        timestamps = timestamps[: args.limit_times]

    print(f"Indexed {len(granules)} complete raw LST granules.")
    print(f"Found {len(groups)} unique overpass timestamps.")
    if missing_rows:
        print(f"Found {len(missing_rows)} granules with missing required layers.")
    if args.limit_times is not None:
        print(f"Processing limited to first {len(timestamps)} timestamps.")
    if args.dry_run:
        return

    reference_path = create_reference_grid()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, str]] = list(missing_rows)
    with rasterio.open(reference_path) as reference:
        if paths["cube"].exists() and args.overwrite:
            paths["cube"].unlink()
        cube = create_cube(paths["cube"], reference, timestamps)
        lst_var = cube.variables["lst"]

        count_total = np.zeros((reference.height, reference.width), dtype="uint16")
        sum_total = np.zeros((reference.height, reference.width), dtype="float64")

        for time_index, timestamp in enumerate(timestamps):
            print(f"[{time_index + 1}/{len(timestamps)}] Processing {timestamp} ({len(groups[timestamp])} tile granules)")
            layer, rows = mosaic_timestamp(groups[timestamp], reference, args)
            summary_rows.extend(rows)
            if layer is None:
                lst_var[time_index, :, :] = np.full((reference.height, reference.width), np.nan, dtype="float32")
                continue
            lst_var[time_index, :, :] = layer
            valid = np.isfinite(layer)
            count_total[valid] += 1
            sum_total[valid] += layer[valid]

        cube.close()

        mean_lst = np.full((reference.height, reference.width), np.nan, dtype="float32")
        valid_count = count_total > 0
        mean_lst[valid_count] = (sum_total[valid_count] / count_total[valid_count]).astype("float32")

        write_raster(paths["count"], reference, count_total, "uint16", None)
        write_raster(paths["mean"], reference, mean_lst, "float32", np.nan)
        write_quicklook(paths["quicklook"], mean_lst)

    write_summary(paths["summary"], summary_rows)
    print(f"Wrote cube: {paths['cube']}")
    print(f"Wrote count raster: {paths['count']}")
    print(f"Wrote mean QA raster: {paths['mean']}")
    print(f"Wrote quicklook: {paths['quicklook']}")
    print(f"Wrote QC summary: {paths['summary']}")


if __name__ == "__main__":
    main()
