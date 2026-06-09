"""Process ECOSTRESS ET and ESI products into Section 3 data cubes.

Section 3 deliverables produced by this script:
- quality-controlled ECOSTRESS ET cube
- quality-controlled ECOSTRESS ESI/PET cube
- usable-observation count rasters for the primary ET and ESI layers
- mean rasters for visual QA only
- per-granule QC summary CSVs
- overpass-level pairing table linking ET granules to matching LST granules

The ET and ESI products do not provide the same LST QC bit field. The QC here
therefore applies the shared cloud and water masks, removes non-finite/fill
values, and removes values outside conservative physical ranges.
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
from rasterio.enums import Resampling
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from create_reference_grid import create_reference_grid


ET_RAW_DIR = config.RAW_DIR / "ecostress_et"
ESI_RAW_DIR = config.RAW_DIR / "ecostress_esi"
LST_RAW_DIR = config.RAW_DIR / "ecostress_lst"
ET_OUTPUT_DIR = config.INTERIM_DIR / "ecostress_et"
ESI_OUTPUT_DIR = config.INTERIM_DIR / "ecostress_esi"

PRODUCT_RE = re.compile(
    r"^ECOv002_(?P<product>.+?)_"
    r"(?P<orbit>\d+)_(?P<scene>\d+)_(?P<tile>\d{2}[A-Z]{3})_"
    r"(?P<timestamp>\d{8}T\d{6})_(?P<processing>\d+)_(?P<instance>\d+)$"
)


@dataclass(frozen=True)
class VariableSpec:
    suffix: str
    name: str
    units: str
    long_name: str
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True)
class ProductConfig:
    label: str
    short_name: str
    raw_dir: Path
    output_dir: Path
    filename_stem: str
    primary_suffix: str
    variables: tuple[VariableSpec, ...]


@dataclass(frozen=True)
class Granule:
    base: str
    product: str
    orbit: str
    scene: str
    tile: str
    timestamp: str
    processing: str
    instance: str
    match_key: str
    paths: dict[str, Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QC-filter, grid, stack, and pair ECOSTRESS ET/ESI products."
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
        help="Process only the first N ET/ESI timestamps. Useful for testing.",
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.001,
        help="Discard a raw granule if the primary layer has less than this valid fraction.",
    )
    parser.add_argument("--et-inst-min", type=float, default=0.0)
    parser.add_argument("--et-inst-max", type=float, default=1000.0)
    parser.add_argument("--et-daily-min", type=float, default=0.0)
    parser.add_argument("--et-daily-max", type=float, default=20.0)
    parser.add_argument("--et-uncertainty-min", type=float, default=0.0)
    parser.add_argument("--et-uncertainty-max", type=float, default=1000.0)
    parser.add_argument("--esi-min", type=float, default=0.0)
    parser.add_argument("--esi-max", type=float, default=1.0)
    parser.add_argument("--pet-min", type=float, default=0.0)
    parser.add_argument("--pet-max", type=float, default=2000.0)
    return parser.parse_args()


def product_configs(args: argparse.Namespace) -> tuple[ProductConfig, ProductConfig]:
    et_config = ProductConfig(
        label="ET",
        short_name="ECO_L3T_JET",
        raw_dir=ET_RAW_DIR,
        output_dir=ET_OUTPUT_DIR,
        filename_stem="ecostress_et",
        primary_suffix="PTJPLSMinst",
        variables=(
            VariableSpec(
                "PTJPLSMinst",
                "et_ptjplsm_inst",
                "native_product_units",
                "Quality-controlled ECOSTRESS PT-JPL-SM instantaneous evapotranspiration",
                args.et_inst_min,
                args.et_inst_max,
            ),
            VariableSpec(
                "ETdaily",
                "et_daily",
                "native_product_units",
                "Quality-controlled ECOSTRESS daily evapotranspiration",
                args.et_daily_min,
                args.et_daily_max,
            ),
            VariableSpec(
                "ETinstUncertainty",
                "et_inst_uncertainty",
                "native_product_units",
                "ECOSTRESS instantaneous evapotranspiration uncertainty",
                args.et_uncertainty_min,
                args.et_uncertainty_max,
            ),
        ),
    )
    esi_config = ProductConfig(
        label="ESI",
        short_name="ECO_L4T_ESI",
        raw_dir=ESI_RAW_DIR,
        output_dir=ESI_OUTPUT_DIR,
        filename_stem="ecostress_esi",
        primary_suffix="ESI",
        variables=(
            VariableSpec(
                "ESI",
                "esi",
                "unitless",
                "Quality-controlled ECOSTRESS evaporative stress index",
                args.esi_min,
                args.esi_max,
            ),
            VariableSpec(
                "PET",
                "pet",
                "native_product_units",
                "Quality-controlled ECOSTRESS potential evapotranspiration",
                args.pet_min,
                args.pet_max,
            ),
        ),
    )
    return et_config, esi_config


def output_paths(product: ProductConfig, limit_times: int | None) -> dict[str, Path]:
    suffix = "" if limit_times is None else f"_first{limit_times}timestamps"
    return {
        "cube": product.output_dir
        / f"{product.filename_stem}_phoenix_20230601_20230930_qc_epsg32612_70m{suffix}.nc",
        "summary": product.output_dir / f"{product.filename_stem}_qc_summary{suffix}.csv",
        "count": product.output_dir
        / f"{product.filename_stem}_usable_observation_count_epsg32612_70m{suffix}.tif",
        "mean": product.output_dir
        / f"{product.filename_stem}_mean_visual_check_epsg32612_70m{suffix}.tif",
    }


def pairing_path(limit_times: int | None) -> Path:
    suffix = "" if limit_times is None else f"_first{limit_times}timestamps"
    return config.INTERIM_DIR / f"ecostress_overpass_pairings{suffix}.csv"


def base_without_suffix(path: Path, suffix: str) -> str:
    marker = f"_{suffix}"
    if not path.stem.endswith(marker):
        raise ValueError(f"{path.name} does not end with {marker}.")
    return path.stem[: -len(marker)]


def parse_base(base: str) -> dict[str, str]:
    match = PRODUCT_RE.match(base)
    if not match:
        raise ValueError(f"Could not parse ECOSTRESS base name: {base}.")
    parts = match.groupdict()
    parts["match_key"] = "_".join(
        [
            parts["orbit"],
            parts["scene"],
            parts["tile"],
            parts["timestamp"],
            parts["instance"],
        ]
    )
    return parts


def index_granules(product: ProductConfig) -> tuple[list[Granule], list[dict[str, str]]]:
    required_suffixes = [spec.suffix for spec in product.variables] + ["cloud", "water"]
    primary_paths = sorted(product.raw_dir.glob(f"*_{product.primary_suffix}.tif"))
    granules: list[Granule] = []
    missing_rows: list[dict[str, str]] = []

    for primary_path in primary_paths:
        base = base_without_suffix(primary_path, product.primary_suffix)
        parts = parse_base(base)
        paths = {suffix: product.raw_dir / f"{base}_{suffix}.tif" for suffix in required_suffixes}
        missing = [suffix for suffix, path in paths.items() if not path.exists()]
        if missing:
            missing_rows.append(
                {
                    "base": base,
                    "product": product.label,
                    "timestamp": parts["timestamp"],
                    "tile": parts["tile"],
                    "match_key": parts["match_key"],
                    "status": "missing_required_files",
                    "reason": ",".join(missing),
                }
            )
            continue
        granules.append(
            Granule(
                base=base,
                product=parts["product"],
                orbit=parts["orbit"],
                scene=parts["scene"],
                tile=parts["tile"],
                timestamp=parts["timestamp"],
                processing=parts["processing"],
                instance=parts["instance"],
                match_key=parts["match_key"],
                paths=paths,
            )
        )

    return granules, missing_rows


def grouped_by_timestamp(granules: Iterable[Granule]) -> dict[str, list[Granule]]:
    groups: dict[str, list[Granule]] = {}
    for granule in granules:
        groups.setdefault(granule.timestamp, []).append(granule)
    return groups


def finite_stats(values: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(values)
    if not finite.any():
        return np.nan, np.nan
    return float(np.nanmin(values[finite])), float(np.nanmax(values[finite]))


def read_float(path: Path) -> tuple[np.ndarray, rasterio.Affine, rasterio.crs.CRS]:
    with rasterio.open(path) as ds:
        data = ds.read(1, masked=True).astype("float32")
        return data.filled(np.nan), ds.transform, ds.crs


def range_mask(values: np.ndarray, spec: VariableSpec) -> np.ndarray:
    mask = np.isfinite(values)
    if spec.minimum is not None:
        mask &= values >= spec.minimum
    if spec.maximum is not None:
        mask &= values <= spec.maximum
    return mask


def read_clean_granule(
    granule: Granule,
    product: ProductConfig,
    min_valid_fraction: float,
) -> tuple[dict[str, np.ndarray] | None, rasterio.Affine | None, rasterio.crs.CRS | None, dict[str, str]]:
    arrays: dict[str, np.ndarray] = {}
    src_transform: rasterio.Affine | None = None
    src_crs: rasterio.crs.CRS | None = None

    for spec in product.variables:
        values, transform, crs = read_float(granule.paths[spec.suffix])
        arrays[spec.name] = values
        if src_transform is None:
            src_transform = transform
            src_crs = crs

    with rasterio.open(granule.paths["cloud"]) as cloud_ds, rasterio.open(
        granule.paths["water"]
    ) as water_ds:
        cloud = cloud_ds.read(1)
        water = water_ds.read(1)

    primary_spec = next(spec for spec in product.variables if spec.suffix == product.primary_suffix)
    primary = arrays[primary_spec.name]
    common_mask = (cloud == 0) & (water == 0)
    primary_mask = common_mask & range_mask(primary, primary_spec)
    valid_fraction = float(primary_mask.sum() / primary_mask.size)

    row = {
        "base": granule.base,
        "product": product.label,
        "timestamp": granule.timestamp,
        "tile": granule.tile,
        "orbit": granule.orbit,
        "scene": granule.scene,
        "instance": granule.instance,
        "match_key": granule.match_key,
        "status": "used",
        "reason": "",
        "source_pixels": str(primary_mask.size),
        "primary_valid_pixels_after_qc": str(int(primary_mask.sum())),
        "primary_valid_fraction_after_qc": f"{valid_fraction:.8f}",
        "cloud_pixels": str(int((cloud != 0).sum())),
        "water_pixels": str(int((water != 0).sum())),
    }

    for spec in product.variables:
        raw_min, raw_max = finite_stats(arrays[spec.name])
        variable_mask = common_mask & range_mask(arrays[spec.name], spec)
        row[f"{spec.name}_raw_min"] = f"{raw_min:.6g}"
        row[f"{spec.name}_raw_max"] = f"{raw_max:.6g}"
        row[f"{spec.name}_valid_pixels_after_qc"] = str(int(variable_mask.sum()))

    if valid_fraction < min_valid_fraction:
        row["status"] = "discarded"
        row["reason"] = "below_min_valid_fraction"
        return None, None, None, row

    cleaned: dict[str, np.ndarray] = {}
    for spec in product.variables:
        variable_mask = common_mask & range_mask(arrays[spec.name], spec)
        cleaned[spec.name] = np.where(variable_mask, arrays[spec.name], np.nan).astype("float32")

    return cleaned, src_transform, src_crs, row


def mosaic_timestamp(
    granules: list[Granule],
    reference: rasterio.DatasetReader,
    product: ProductConfig,
    min_valid_fraction: float,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    sums = {
        spec.name: np.zeros((reference.height, reference.width), dtype="float64")
        for spec in product.variables
    }
    counts = {
        spec.name: np.zeros((reference.height, reference.width), dtype="uint16")
        for spec in product.variables
    }
    rows: list[dict[str, str]] = []

    for granule in granules:
        cleaned, src_transform, src_crs, row = read_clean_granule(
            granule, product, min_valid_fraction
        )
        rows.append(row)
        if cleaned is None or src_transform is None or src_crs is None:
            continue

        for spec in product.variables:
            dest = np.full((reference.height, reference.width), np.nan, dtype="float32")
            reproject(
                source=cleaned[spec.name],
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
            sums[spec.name][valid] += dest[valid]
            counts[spec.name][valid] += 1

    layers: dict[str, np.ndarray] = {}
    for spec in product.variables:
        layer = np.full((reference.height, reference.width), np.nan, dtype="float32")
        valid = counts[spec.name] > 0
        layer[valid] = (sums[spec.name][valid] / counts[spec.name][valid]).astype("float32")
        layers[spec.name] = layer

    return layers, rows


def create_cube(
    path: Path,
    reference: rasterio.DatasetReader,
    timestamps: list[str],
    product: ProductConfig,
) -> netCDF4.Dataset:
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

    for spec in product.variables:
        var = ds.createVariable(
            spec.name,
            "f4",
            ("time", "y", "x"),
            zlib=True,
            complevel=4,
            chunksizes=(1, min(256, reference.height), min(256, reference.width)),
            fill_value=np.nan,
        )
        var.units = spec.units
        var.long_name = spec.long_name

    ds.title = f"ECOSTRESS {product.short_name} quality-controlled Phoenix {product.label} cube"
    ds.source_product = product.short_name
    ds.crs = str(reference.crs)
    ds.grid_cell_size_m = str(config.GRID_CELL_SIZE_M)
    ds.transform = ",".join(str(value) for value in reference.transform.to_gdal())
    ds.history = f"Created {datetime.now(timezone.utc).isoformat()}"
    ds.note = (
        "QC mask removes cloud != 0, water != 0, non-finite/fill values, "
        "and values outside product-specific plausible ranges. Native product "
        "GeoTIFFs do not expose the LST QC bit field."
    )
    return ds


def write_raster(
    path: Path,
    reference: rasterio.DatasetReader,
    data: np.ndarray,
    dtype: str,
    nodata,
) -> None:
    profile = reference.profile.copy()
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(dtype), 1)


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_product(
    product: ProductConfig,
    reference: rasterio.DatasetReader,
    args: argparse.Namespace,
) -> tuple[list[Granule], list[str], dict[str, Path]]:
    paths = output_paths(product, args.limit_times)
    if paths["cube"].exists() and not args.overwrite and not args.dry_run:
        raise SystemExit(f"Output exists: {paths['cube']}. Use --overwrite to recreate it.")

    granules, missing_rows = index_granules(product)
    groups = grouped_by_timestamp(granules)
    timestamps = sorted(groups)
    if args.limit_times is not None:
        timestamps = timestamps[: args.limit_times]

    print(f"{product.label}: indexed {len(granules)} complete raw granules.")
    print(f"{product.label}: found {len(groups)} unique overpass timestamps.")
    if missing_rows:
        print(f"{product.label}: found {len(missing_rows)} granules with missing required layers.")
    if args.limit_times is not None:
        print(f"{product.label}: processing limited to first {len(timestamps)} timestamps.")
    if args.dry_run:
        return granules, timestamps, paths

    product.output_dir.mkdir(parents=True, exist_ok=True)
    if paths["cube"].exists() and args.overwrite:
        paths["cube"].unlink()

    cube = create_cube(paths["cube"], reference, timestamps, product)
    count_total = np.zeros((reference.height, reference.width), dtype="uint16")
    sum_primary = np.zeros((reference.height, reference.width), dtype="float64")
    summary_rows: list[dict[str, str]] = list(missing_rows)
    primary_var_name = next(
        spec.name for spec in product.variables if spec.suffix == product.primary_suffix
    )

    for time_index, timestamp in enumerate(timestamps):
        print(
            f"{product.label} [{time_index + 1}/{len(timestamps)}] "
            f"Processing {timestamp} ({len(groups[timestamp])} tile granules)"
        )
        layers, rows = mosaic_timestamp(
            groups[timestamp], reference, product, args.min_valid_fraction
        )
        summary_rows.extend(rows)

        for spec in product.variables:
            cube.variables[spec.name][time_index, :, :] = layers[spec.name]

        primary_layer = layers[primary_var_name]
        valid = np.isfinite(primary_layer)
        count_total[valid] += 1
        sum_primary[valid] += primary_layer[valid]

    cube.close()

    mean_primary = np.full((reference.height, reference.width), np.nan, dtype="float32")
    valid_count = count_total > 0
    mean_primary[valid_count] = (sum_primary[valid_count] / count_total[valid_count]).astype(
        "float32"
    )
    write_raster(paths["count"], reference, count_total, "uint16", None)
    write_raster(paths["mean"], reference, mean_primary, "float32", np.nan)
    write_summary(paths["summary"], summary_rows)

    print(f"{product.label}: wrote cube: {paths['cube']}")
    print(f"{product.label}: wrote count raster: {paths['count']}")
    print(f"{product.label}: wrote mean QA raster: {paths['mean']}")
    print(f"{product.label}: wrote QC summary: {paths['summary']}")
    return granules, timestamps, paths


def index_pairable(raw_dir: Path, primary_suffix: str) -> dict[str, Granule]:
    product = ProductConfig(
        label=primary_suffix,
        short_name="",
        raw_dir=raw_dir,
        output_dir=raw_dir,
        filename_stem="",
        primary_suffix=primary_suffix,
        variables=(VariableSpec(primary_suffix, primary_suffix, "", "", None, None),),
    )
    granules, _ = index_granules_without_masks(product)
    return {granule.match_key: granule for granule in granules}


def index_granules_without_masks(product: ProductConfig) -> tuple[list[Granule], list[dict[str, str]]]:
    primary_paths = sorted(product.raw_dir.glob(f"*_{product.primary_suffix}.tif"))
    granules: list[Granule] = []
    missing_rows: list[dict[str, str]] = []
    for primary_path in primary_paths:
        base = base_without_suffix(primary_path, product.primary_suffix)
        parts = parse_base(base)
        granules.append(
            Granule(
                base=base,
                product=parts["product"],
                orbit=parts["orbit"],
                scene=parts["scene"],
                tile=parts["tile"],
                timestamp=parts["timestamp"],
                processing=parts["processing"],
                instance=parts["instance"],
                match_key=parts["match_key"],
                paths={product.primary_suffix: primary_path},
            )
        )
    return granules, missing_rows


def write_pairing_table(
    path: Path,
    et_granules: list[Granule],
    et_timestamps: list[str],
    esi_timestamps: list[str],
) -> None:
    lst_by_key = index_pairable(LST_RAW_DIR, "LST")
    esi_by_key = index_pairable(ESI_RAW_DIR, "ESI")
    et_time_index = {timestamp: index for index, timestamp in enumerate(et_timestamps)}
    esi_time_index = {timestamp: index for index, timestamp in enumerate(esi_timestamps)}
    lst_timestamps = sorted({granule.timestamp for granule in lst_by_key.values()})
    lst_time_index = {timestamp: index for index, timestamp in enumerate(lst_timestamps)}
    processed_et_timestamps = set(et_timestamps)

    rows: list[dict[str, str]] = []
    for et in sorted(et_granules, key=lambda granule: granule.match_key):
        if et.timestamp not in processed_et_timestamps:
            continue
        lst = lst_by_key.get(et.match_key)
        esi = esi_by_key.get(et.match_key)
        status_parts = ["et"]
        if lst is not None:
            status_parts.append("lst")
        if esi is not None:
            status_parts.append("esi")
        rows.append(
            {
                "match_key": et.match_key,
                "timestamp": et.timestamp,
                "orbit": et.orbit,
                "scene": et.scene,
                "tile": et.tile,
                "instance": et.instance,
                "pairing_status": "_".join(status_parts),
                "has_matching_lst": str(lst is not None).lower(),
                "has_matching_esi": str(esi is not None).lower(),
                "et_base": et.base,
                "lst_base": "" if lst is None else lst.base,
                "esi_base": "" if esi is None else esi.base,
                "et_primary_path": et.paths["PTJPLSMinst"].relative_to(config.PROJECT_ROOT).as_posix(),
                "lst_primary_path": ""
                if lst is None
                else lst.paths["LST"].relative_to(config.PROJECT_ROOT).as_posix(),
                "esi_primary_path": ""
                if esi is None
                else esi.paths["ESI"].relative_to(config.PROJECT_ROOT).as_posix(),
                "et_cube_time_index": str(et_time_index.get(et.timestamp, "")),
                "lst_cube_time_index": str(lst_time_index.get(et.timestamp, "")),
                "esi_cube_time_index": str(esi_time_index.get(et.timestamp, "")),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "match_key",
        "timestamp",
        "orbit",
        "scene",
        "tile",
        "instance",
        "pairing_status",
        "has_matching_lst",
        "has_matching_esi",
        "et_base",
        "lst_base",
        "esi_base",
        "et_primary_path",
        "lst_primary_path",
        "esi_primary_path",
        "et_cube_time_index",
        "lst_cube_time_index",
        "esi_cube_time_index",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    matched_lst = sum(1 for row in rows if row["has_matching_lst"] == "true")
    matched_esi = sum(1 for row in rows if row["has_matching_esi"] == "true")
    print(f"Wrote pairing table: {path}")
    print(f"Pairing table ET rows: {len(rows)}")
    print(f"ET rows with matching LST granule: {matched_lst}")
    print(f"ET rows with matching ESI granule: {matched_esi}")


def main() -> None:
    args = parse_args()
    et_config, esi_config = product_configs(args)

    reference_path = create_reference_grid()
    with rasterio.open(reference_path) as reference:
        et_granules, et_timestamps, _ = process_product(et_config, reference, args)
        esi_granules, esi_timestamps, _ = process_product(esi_config, reference, args)

    if args.dry_run:
        return

    pair_path = pairing_path(args.limit_times)
    if pair_path.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {pair_path}. Use --overwrite to recreate it.")
    write_pairing_table(pair_path, et_granules, et_timestamps, esi_timestamps)


if __name__ == "__main__":
    main()
