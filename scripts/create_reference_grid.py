"""Create the fixed 70 m Phoenix reference raster for all aligned layers."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config


def reference_grid_path() -> Path:
    return config.INTERIM_DIR / "reference_grid_phoenix_epsg32612_70m.tif"


def projected_bounds() -> tuple[float, float, float, float]:
    lon_min, lon_max, lat_min, lat_max = config.PHOENIX_BBOX
    transformer = Transformer.from_crs("EPSG:4326", config.PROJECTED_CRS, always_xy=True)
    corners = [
        transformer.transform(lon_min, lat_min),
        transformer.transform(lon_min, lat_max),
        transformer.transform(lon_max, lat_min),
        transformer.transform(lon_max, lat_max),
    ]
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return min(xs), min(ys), max(xs), max(ys)


def create_reference_grid(path: Path | None = None, overwrite: bool = False) -> Path:
    output_path = path or reference_grid_path()
    if output_path.exists() and not overwrite:
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    min_x, min_y, max_x, max_y = projected_bounds()
    cell_size = config.GRID_CELL_SIZE_M

    origin_x = math.floor(min_x / cell_size) * cell_size
    origin_y = math.ceil(max_y / cell_size) * cell_size
    width = math.ceil((max_x - origin_x) / cell_size)
    height = math.ceil((origin_y - min_y) / cell_size)
    transform = from_origin(origin_x, origin_y, cell_size, cell_size)

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": config.PROJECTED_CRS,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "nodata": 255,
    }
    data = np.zeros((height, width), dtype=np.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(
            purpose="reference_grid",
            pilot_city=config.PILOT_CITY,
            bbox_lonlat=",".join(str(value) for value in config.PHOENIX_BBOX),
            grid_cell_size_m=str(config.GRID_CELL_SIZE_M),
        )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the fixed Phoenix reference grid.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the grid if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = create_reference_grid(overwrite=args.overwrite)
    with rasterio.open(path) as ds:
        print(f"Reference grid: {path}")
        print(f"CRS: {ds.crs}")
        print(f"Shape: {ds.height} rows x {ds.width} columns")
        print(f"Cell size: {ds.transform.a} m")


if __name__ == "__main__":
    main()
