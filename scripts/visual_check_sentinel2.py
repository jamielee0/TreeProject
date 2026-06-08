"""Create a quick visual QA PNG for the Sentinel-2 NDVI composite.

This avoids Matplotlib because that stack can crash in some Windows conda
environments. The output is a simple NDVI quicklook with labeled spot checks.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from pyproj import Transformer


NDVI_PATH = Path("data/interim/sentinel2/sentinel2_phoenix_20230601_20230930_ndvi_median_epsg32612_70m.tif")
FIGURE_PATH = Path("figures/sentinel2_ndvi_visual_check.png")
CSV_PATH = Path("figures/sentinel2_ndvi_spotcheck.csv")

SAMPLE_POINTS = [
    ("Encanto Park", 33.4776, -112.0908, "green"),
    ("Steele Indian School Park", 33.4955, -112.0708, "green"),
    ("Downtown Phoenix", 33.4484, -112.0740, "built"),
    ("Sky Harbor runways", 33.4343, -112.0116, "built"),
    ("Industrial west Phoenix", 33.4430, -112.1500, "built"),
]


def ndvi_to_rgb(ndvi: np.ndarray) -> np.ndarray:
    """Map NDVI values to a brown-white-green RGB ramp."""
    clipped = np.clip((ndvi + 0.1) / 0.9, 0.0, 1.0)
    clipped = np.where(np.isfinite(clipped), clipped, 0.0)
    brown = np.array([120, 85, 55], dtype=np.float32)
    white = np.array([238, 236, 220], dtype=np.float32)
    green = np.array([20, 120, 45], dtype=np.float32)

    rgb = np.empty((*clipped.shape, 3), dtype=np.float32)
    low = clipped <= 0.5
    high = ~low
    rgb[low] = brown + (white - brown) * (clipped[low, None] / 0.5)
    rgb[high] = white + (green - white) * ((clipped[high, None] - 0.5) / 0.5)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_marker(draw: ImageDraw.ImageDraw, x: int, y: int, kind: str, label: str) -> None:
    if kind == "green":
        fill = (0, 210, 255)
        outline = (255, 255, 255)
        radius = 5
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill, outline=outline, width=2)
    else:
        fill = (0, 0, 0)
        size = 6
        draw.line((x - size, y - size, x + size, y + size), fill=fill, width=3)
        draw.line((x - size, y + size, x + size, y - size), fill=fill, width=3)
    draw.rectangle((x + 8, y + 6, x + 8 + len(label) * 6 + 4, y + 20), fill=(255, 255, 255))
    draw.text((x + 10, y + 7), label, fill=(0, 0, 0))


def main() -> None:
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32612", always_xy=True)

    with rasterio.open(NDVI_PATH) as ds:
        ndvi = ds.read(1).astype("float32")
        ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan)
        step = 4
        preview = ndvi[::step, ::step]
        rgb = ndvi_to_rgb(preview)
        image = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(image)

        rows = []
        for name, lat, lon, kind in SAMPLE_POINTS:
            x, y = transformer.transform(lon, lat)
            row, col = ds.index(x, y)
            window = ndvi[max(0, row - 1) : row + 2, max(0, col - 1) : col + 2]
            median_3x3 = float(np.nanmedian(window))
            center = float(ndvi[row, col])
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "lat": lat,
                    "lon": lon,
                    "row": row,
                    "col": col,
                    "ndvi_center": center,
                    "ndvi_3x3_median": median_3x3,
                }
            )
            draw_marker(draw, int(col / step), int(row / step), kind, name)

    image.save(FIGURE_PATH)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {FIGURE_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
