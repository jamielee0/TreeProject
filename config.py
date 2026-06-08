"""Project configuration for the Phoenix urban tree threshold pilot.

Keep fixed study constants here so every script and notebook uses the same
domain, projection, grid, and time windows.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"


# Phoenix pilot domain: lon_min, lon_max, lat_min, lat_max.
PHOENIX_BBOX = (-112.55, -111.55, 33.20, 33.92)

# All analysis layers should be reprojected and snapped to this grid.
PROJECTED_CRS = "EPSG:32612"
GRID_CELL_SIZE_M = 70


PILOT_CITY = "Phoenix"
PILOT_ANALYSIS_START = "2023-06-01"
PILOT_ANALYSIS_END = "2023-09-30"

WARM_SEASON_START_MONTH_DAY = "06-01"
WARM_SEASON_END_MONTH_DAY = "09-30"
CLIMATOLOGY_START_YEAR = 2018
CLIMATOLOGY_END_YEAR = 2024


# Initial pixel-classification thresholds from the protocol.
TREE_NDVI_MIN = 0.5
TREE_CANOPY_FRACTION_MIN = 0.70
TREE_IMPERVIOUS_FRACTION_MAX = 0.20
MIN_GOOD_ECOSTRESS_OBSERVATIONS = 20
BUILDING_BUFFER_M = 70
