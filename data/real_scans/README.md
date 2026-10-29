# Real scan data (not committed)

Everything in this folder except this file is ignored by git (see `.gitignore`): the volumes are tens of
gigabytes. Keep a backup copy elsewhere if you need one - git will not hold it.

## `fragments/` - official Vesuvius Challenge ink-detection fragments

| Folder | Source fragment | Slices (`surface_volume/`) | Files |
|---|---|---|---|
| `frag1/` | PHercParis2Fr47 | 65 TIFFs, ~99 MiB each | + `inklabels.png`, `mask.png` |
| `frag2/` | PHercParis2Fr143 | 65 TIFFs, ~269 MiB each | + `inklabels.png`, `mask.png` |
| `frag3/` | PHercParis1Fr34 | 65 TIFFs, ~76 MiB each | + `inklabels.png`, `mask.png` |

Total: 201 files, about 28.2 GiB.

- **Source:** https://dl.ash2txt.org/fragments/ - public, no login. Each file comes from
  `<FragN>/<volpkg>/working/54keV_exposed_surface/` (`surface_volume/NN.tif`, `inklabels.png`, `mask.png`).
- **Scan:** 54 keV synchrotron micro-CT, 3.24 um voxels. `inklabels.png` is the ground-truth ink
  (binary); `mask.png` marks which pixels are papyrus.
- **Licence / attribution:** data published by the Vesuvius Challenge (scrollprize.org); check their
  data licence and cite them if you publish results.

### Download / resume

The downloader is resumable and skips files that are already complete (sizes are checked against the server):

```bash
.venv/Scripts/python.exe -u scripts/download_fragments.py --out data/real_scans/fragments --fragments 1 2 3 --workers 12
```

`./run_download_then_train.sh` does the download and then trains the ink model.

### Derived files (safe to delete, rebuilt on demand)

- `fragN/stack_z24_n16.npy` - slices 24-39 decoded into one uint16 array for fast training.
- `fragN/surface_volume/*.part` - a half-finished download; re-running the downloader completes it.

Trained models and results go to `data/processed/models/` (also ignored by git).

## `pherc0332_native_crop.h5` - native-resolution scroll crop

- 1024 x 2048 x 2048 uint8 voxels (4.0 GiB) at 2.399 um, `volume` dataset; attributes record the source and offsets.
- Source: `s3://vesuvius-challenge-open-data/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr`, level 0,
  z 16256-17280, y and x 7168-9216 (the scroll's centre).
- Fetch: `python scripts/download_native_crop.py --out data/real_scans/pherc0332_native_crop.h5 --z0 16256 --y0 7168 --x0 7168 --depth 1024 --size 2048`
- `pherc0332_native_tile_corrected.h5` is a derived float32 768 x 1024 x 1024 tile (uint8/255, M1 skipped) - safe to delete.

## `paris4/` - labelled Scroll 1 (PHerc. Paris 4) tile for the ink evaluation

Used by `scripts/eval_ink_labels.py` (see `docs/INK_EVALUATION.md`). All public, anonymous.

| File | Source | Size |
|---|---|---|
| `w00/` (`x.tif`, `y.tif`, `z.tif`, `meta.json`, `w00_20231016151002_inklabels.tif`, `..._supervision_mask.tif`) | `https://huggingface.co/buckets/scrollprize/datasets/resolve/ink/phercparis4/w00_20231016151002/<file>` | ~220 MB |
| `w01/` labels + geometry (`w01_20230702185753_r15_inklabels.tif`, `..._supervision_mask.tif`, `x/y/z.tif`, `meta.json`), and per-tile `native_crop.h5`, `surface_pred_crop.h5`, `tile.json`, derived `tile.h5` | labels: same bucket as `w00`, folder `phercparis4/w01_*` (exact folder name not recorded at download time); crop z 69760 y 22016 x 17408, 1536 x 1152 x 1152 | 1.90 GiB crop |
| `w02/` labels + geometry (`w02_20231031143852_inklabels.tif`, `..._supervision_mask.tif`, `x/y/z.tif`, `meta.json`), and per-tile `native_crop.h5`, `surface_pred_crop.h5`, `tile.json`, derived `tile.h5` | labels: same bucket as `w00`, folder `phercparis4/w02_*` (exact folder name not recorded at download time); crop z 38912 y 11648 x 18560, 1280 x 896 x 1152 | 1.23 GiB crop |
| `w03/` labels + geometry (`w03_20231106155351_inklabels.tif`, `..._supervision_mask.tif`, `x/y/z.tif`, `meta.json`), and per-tile `native_crop.h5`, `surface_pred_crop.h5`, `tile.json`, derived `tile.h5` | labels: same bucket as `w00`, folder `phercparis4/w03_*` (exact folder name not recorded at download time); crop z 52864 y 16256 x 18944, 1408 x 768 x 1280 | 1.29 GiB crop |
| `w05/` labels + geometry (`w05_4424_inklabels.tif`, `..._supervision_mask.tif`, `x/y/z.tif`, `meta.json`), and per-tile `native_crop.h5`, `surface_pred_crop.h5`, `tile.json`, derived `tile.h5` | labels: same bucket as `w00`, folder `phercparis4/w05_*` (exact folder name not recorded at download time); crop z 42624 y 15488 x 19072, 1024 x 1536 x 1408 | 2.06 GiB crop |
| `native_crop.h5` (1536 x 1408 x 768 uint8 at 2.4 um, z 62080 y 17280 x 18816) | S3 `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | 1.55 GB |
| `surface_pred_crop.h5` (`surface-m7-L2-th0.2`, scale 4) | S3 `PHercParis4/representations/predictions/surfaces/` | 25 MB |
| `umbilicus.json` | S3 `PHercParis4/representations/umbilicus/20260411134726-umbilicus-20260524235033.json` | 11 KB |
| `tile.h5` (derived: `native_crop.h5` as float32/255, 6.6 GB; safe to delete) | built by the pipeline | 6.6 GB |

```bash
python scripts/download_native_crop.py --out data/real_scans/paris4/native_crop.h5 --scroll-id PHercParis4 --volume 2.400um-0.2m-78keV --voxel-um 2.4     --z0 62080 --y0 17280 --x0 18816 --depth 1536 --ysize 1408 --xsize 768
python scripts/download_surface_prediction.py --out data/real_scans/paris4/surface_pred_crop.h5 --scroll-id PHercParis4 --name-contains m7-L2     --z0 15520 --y0 4320 --x0 4704 --depth 384 --ysize 352 --xsize 192 --scale 4
# tiles for w01 w02 w03 w05 (chosen with scripts/pick_label_tile.py; surface-prediction crop = same box / 4 with --scale 4)
python scripts/download_native_crop.py --out data/real_scans/paris4/w01/native_crop.h5 --scroll-id PHercParis4 --volume 2.400um-0.2m-78keV --voxel-um 2.4 --z0 69760 --y0 22016 --x0 17408 --depth 1536 --ysize 1152 --xsize 1152
python scripts/download_native_crop.py --out data/real_scans/paris4/w02/native_crop.h5 --scroll-id PHercParis4 --volume 2.400um-0.2m-78keV --voxel-um 2.4 --z0 38912 --y0 11648 --x0 18560 --depth 1280 --ysize 896 --xsize 1152
python scripts/download_native_crop.py --out data/real_scans/paris4/w03/native_crop.h5 --scroll-id PHercParis4 --volume 2.400um-0.2m-78keV --voxel-um 2.4 --z0 52864 --y0 16256 --x0 18944 --depth 1408 --ysize 768 --xsize 1280
python scripts/download_native_crop.py --out data/real_scans/paris4/w05/native_crop.h5 --scroll-id PHercParis4 --volume 2.400um-0.2m-78keV --voxel-um 2.4 --z0 42624 --y0 15488 --x0 19072 --depth 1024 --ysize 1536 --xsize 1408
```
The label geometry places the segment in the 2.4 um volume's frame, but it sits ~8 voxels off the papyrus there; score only after refining (the evaluation does).

## `pherc0211/` - PHerc. 0211, scan `20250821151803` (Vesuvius Challenge First Letters / Grand Prize target)

Eligible for the 2027 Grand Prize and the First Letters Prize (deadline 25 June 2027, 11:59 pm Pacific; see https://scrollprize.org/prizes).
**Scan ID `20250821151803`** must appear in every record, render and methodology for this scroll. ESRF, 113 keV, 9.362 um voxels, CC BY-NC 4.0.
As of 2026-10-10 the data browser lists **no segments and no ink predictions** for this scroll.

| Path | Source (S3 `vesuvius-challenge-open-data/PHerc0211/...`) | Size |
|---|---|---|
| `volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/{1..5}` | `volumes/<same name>` (level 0 is 19416 x 7948 x 7948, ~300 GB compressed, not mirrored) | level 1 = 38 GB, level 2 = 5.9 GB, 3-5 = 1.3 GB |
| `representations/umbilicus/20250821151803-umbilicus-20260808112626.json` | `representations/umbilicus/` | 4 KB |
| `representations/predictions/surfaces/...surface-m7-L0-th0.2.zarr/{2..5}` | `representations/predictions/surfaces/` (levels 0-1 not mirrored) | 0.66 GB |
| `photos/` | `photos/PHerc0211_photo.jpg`, `PHerc0211_mask.png` | 8 MB |
| `native_crop_A.h5` | level 0, z 9600-9856, y 3840-4480, x 4608-6144, uint8 | 0.4 GB |

```bash
python scripts/download_s3_prefix.py --out data/real_scans/pherc0211 --strip PHerc0211/ \
    --prefix PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr/1 --prefix PHerc0211/representations/umbilicus
python scripts/download_native_crop.py --out data/real_scans/pherc0211/native_crop_A.h5 --scroll-id PHerc0211 --volume 9.362um \
    --voxel-um 9.362 --z0 9600 --y0 3840 --x0 4608 --depth 256 --ysize 640 --xsize 1536
```
