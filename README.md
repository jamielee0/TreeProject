# Urban Tree Thermal Thresholds

Repository for the Phoenix pilot of the urban canopy thermal-threshold study.

The repo is organized so raw data, intermediate rasters, processed analysis
tables, notebooks, reusable code, and figures stay separate. Large downloaded
or generated data files should not be committed to Git. Record every acquired
file in `manifest.csv`.

## Structure

- `config.py` stores fixed study constants such as domain, CRS, grid size, and time windows.
- `data/raw/` stores downloaded source files that are never edited by hand.
- `data/interim/` stores aligned or quality-controlled intermediate layers.
- `data/processed/` stores final analysis-ready datasets.
- `manifest.csv` tracks source files, dates, and checksums.
- `src/` stores reusable project code.
- `notebooks/` stores numbered exploratory notebooks.
- `figures/` stores generated figures and maps.
