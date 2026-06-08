# Scripts

Command-line scripts for project setup and data acquisition.

Run scripts from the repository root after activating the project environment:

```powershell
conda activate urban-tree-thresholds
python scripts/download_ecostress_lst.py --dry-run
python scripts/process_ecostress_lst.py --dry-run
python scripts/download_ecostress_et.py --dry-run
python scripts/download_ecostress_esi.py --dry-run
python scripts/download_sentinel2_indices.py --dry-run
python scripts/create_reference_grid.py
python scripts/download_landcover_layers.py --dry-run
```

Raw ECOSTRESS downloads are saved under:

- `data/raw/ecostress_lst/` for `ECO_L2T_LSTE`
- `data/raw/ecostress_et/` for `ECO_L3T_JET`
- `data/raw/ecostress_esi/` for `ECO_L4T_ESI`

Processed ECOSTRESS LST outputs are saved under:

- `data/interim/ecostress_lst/` for the QC-filtered LST cube, count raster, mean QA raster, and QC summary
- `figures/ecostress_lst_mean_visual_check.png` for a mean-LST visual QA map only

Sentinel-2 index composites are saved under:

- `data/interim/sentinel2/` for warm-season median `NDVI` and `NDMI`

Land-cover layers are saved under:

- `data/raw/landcover/` for native 30 m Earth Engine exports
- `data/interim/landcover/` for 70 m aligned impervious fraction, tree-canopy fraction, and NLCD land-cover class
