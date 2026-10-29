# Ink detection against real ground-truth labels (PHerc. Paris 4 / Scroll 1)

> **CORRECTION (2026-10-12): the conclusion below that the model has "real but modest signal" on Scroll 1 is RETRACTED.** A spatial-shift null (`scripts/probes/ink_spatial_null.py`; results in `results/paris4_spatial_null/`) shows the AUCs reported here are not distinguishable from chance: observed vs null 5-95% - w00 0.653 vs 0.23-0.79 (p 0.36), w01 0.455 vs 0.37-0.66 (p 0.60), w02 0.362 vs 0.12-0.78 (p 0.47), w03 0.542 vs 0.12-0.74 (p 0.84). The labels are a few large blobs and the model output is smooth blobs, so shifting the labels by hundreds of voxels gives AUC anywhere from 0.02 to 0.98 (null sizes are small on w02, n=14). The depth-scan swings and the "0.08 AUC gap" investigation below are inside the same null range. **Update 2026-10-23:** `w05` is also chance (0.524, p 0.87), and the more powerful edge-band test finds no signal on any of the five segments; see "Five segments, spatial-shift null and edge-band test" below. **Update 2026-10-29:** the edge-band result is NOT evidence that no ink was present: the official in-sample Paris 4 3D model (`scrollprize/ink_3d_dino_guided`) scores 0.70-0.88 pooled (p <= 0.03 vs the shifted-label null) on all five segments but is itself at chance on most edge-band and high-passed tests (`docs/LITERATURE_GAP_ANALYSIS.md`, `results/ink3d_official/`), so the edge-band test cannot detect a working ink model on these labels. The chance results here come from our own fragment U-Net. The rest of this note is kept for the record and has not been rewritten.

Until now no ink result in this project could be checked: PHerc. 0332 has no ink labels. This evaluation scores the fragment-trained U-Net (`ink_unet_20k.pt`, held-out fragment F0.5 0.52)
against the official ink labels of a Scroll 1 segment, on a patch scanned the same way as 0332 (2.400 um, 78 keV vs 2.399 um, 78 keV), through this project's own M2 -> M3 -> ink pipeline.

## Data (all public; nothing under `data/` is committed)

| item | source | size |
|---|---|---|
| ink labels + supervision mask of segment `w00_20231016151002`, its `x/y/z.tif` geometry | Hugging Face bucket `scrollprize/datasets`, `ink/phercparis4/w00_20231016151002/` | ~220 MB |
| CT crop, z 62080-63616, y 17280-18688, x 18816-19584 (1536 x 1408 x 768) | S3 `vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | 1.55 GB |
| surface-prediction crop (`surface-m7-L2-th0.2`, 4x coarser) | S3 `.../PHercParis4/representations/predictions/surfaces/` | 25 MB |
| scroll axis | S3 `.../PHercParis4/representations/umbilicus/20260411134726-umbilicus-20260524235033.json` | 11 KB |

Four other labelled Paris 4 segments exist (`w01`, `w02`, `w03`, `w05`); their label images and geometry are 205-430 MB each, but each segment's surface-volume zarr is 77-108 GB and was not downloaded.
The tile was chosen where the labels are densest (a 1024 x 1024-pixel block that is fully supervised and 50 % ink), with a 64-voxel margin. Commands: see `data/real_scans/README.md`.

## Method (`scripts/eval_ink_labels.py`)

A label is a 2D image aligned to the segment; the segment's `x/y/z.tif` grid places each label pixel in 3D, in the scroll volume's own frame (checked: the segment lies inside the 2.4 um volume). Only pixels in the **supervision mask** are scored. 3.01 million supervised label
pixels fall inside the tile, 29.2 % of them ink. An ink map is scored at those points: each label point takes the ink probability of the nearest page pixel within `--dmax` voxels (8 by default), so any flattening is scored on the same labels. Coverage = the share of
label pixels a flattening reaches. Metrics: AUC, best-threshold F0.5, and the trivial "everything is ink" F0.5. `--gold` also runs the model on the segment's *own* surface (the page the labels were drawn on). Unit tests: `tests/test_eval_ink_labels.py`.

## Results

All scores are for the model's reading with the page depth in the "normal" direction; the flipped direction is at chance for every method (AUC 0.50-0.54), so the model's signal depends on reading the stack the right way up.

**On exactly the same label pixels** (the 1.07 M supervised pixels, 35.5 %, that all three methods reach; 45.7 % ink; "all ink" F0.5 = 0.457, chance AUC = 0.5):

| pages | AUC | best F0.5 | precision / recall |
|---|---|---|---|
| segment's own surface, refined onto the papyrus ("gold") | **0.729** | **0.657** | 0.69 / 0.54 |
| this pipeline, plain M2 pages (baseline) | 0.654 | 0.540 | 0.54 / 0.56 |
| this pipeline, split pages | 0.652 | 0.539 | 0.54 / 0.55 |

Each method on its own reachable labels (`dmax` 8 voxels): gold + refine AUC 0.721 / F0.5 0.635 (60.6 % of labels reached; all-ink 0.420), baseline 0.655 / 0.540 (37.1 %), split 0.653 / 0.538 (37.2 %). Un-refined gold geometry: AUC 0.557 / F0.5 0.356 (all 3.0 M labels; all-ink 0.340).
Sensitivity of the pipeline pages to the gate (normal depth, baseline / split AUC): 0.666 / 0.664 at 4 voxels (16 % coverage), 0.655 / 0.653 at 8 (37 %), 0.631 / 0.630 at 12 (59 %), 0.622 / 0.622 at 16 (77 %).

Figure: [`figures/paris4_labels_vs_model.png`](figures/paris4_labels_vs_model.png) - one split page: CT | model ink probability | ground truth (yellow ink, blue no ink, grey unlabelled).
Merged pages for this tile: [`figures/paris4_merged_page_split.png`](figures/paris4_merged_page_split.png), [`figures/paris4_overview_split.png`](figures/paris4_overview_split.png). Raw numbers: [`ink_eval/`](ink_eval/).

## What this shows

1. **The model has real but modest signal on Scroll 1.** AUC 0.65 on this pipeline's pages and 0.73 on the segment's own surface; best F0.5 0.54 / 0.66 against 0.46 for labelling everything ink. It is far from a reliable detector.
   The ground truth here is blobby (large irregular regions, not letter strokes) and the model's output is smooth blobs, so this says the *regions* roughly agree; it says nothing about reading letters.
2. **Surface alignment decides most of the score.** The official segment sits about 8 voxels (~20 um) off the papyrus ridge in the 2.4 um volume (a 1-voxel offset on the older ~8 um scan the labels were drawn on is ~3 voxels here): only 33 % of its vertices are on predicted surface and the raw CT
   rises monotonically along the normal instead of peaking at the mesh. Fed directly, the model scores AUC 0.557 (near chance); after `refine_surface` snaps it onto the papyrus, 0.721. **An earlier version of this analysis concluded the model "barely beats chance" from the un-refined run; that was wrong, the cause was the misaligned geometry.**
3. **Splitting wraps makes no difference here, and this tile cannot show that it would.** Baseline and split pages reach the same label pixels (37.1 % vs 37.2 %) and score the same (0.654 vs 0.652): the labelled sheet is a single wrap that both pipelines flatten identically.
   The gain from splitting was on pages that crossed wraps (see [WRAP_SPLITTING.md](WRAP_SPLITTING.md)), and none of those overlap this label region. A test of that claim needs labels on a region whose baseline pages are multi-wrap.
4. **This pipeline's pages lose about 0.08 AUC against the official segment's geometry** (0.65 vs 0.73 on common pixels) even though both are refined. Investigated below: partly the refine step, not fully explained.

## Why the pipeline's pages trail the official surface (investigation)

Scripts in `scripts/probes/`. All numbers: Paris 4 tile, `w00` labels, split pipeline pages, normal depth.

**Ruled out**
* *Page orientation.* The sign of a page's normal (which side is "up") is arbitrary per page, so a global "normal" run could mix orientations. It does not: every large page has the same outward sign as the official surface, and per-page radially-consistent orientation (0.651) or an oracle choosing the best orientation per page (0.654) equals the global setting (0.654).
* *Label-to-page distance.* The gap is the same in every distance bin: 0.672 vs 0.738 at 0-2 voxels, 0.658 vs 0.731 at 2-4, 0.661 vs 0.735 at 4-6, 0.636 vs 0.711 at 6-8.
* *A constant depth offset.* See the depth scan: the two surfaces agree where the stack is shifted one way and diverge the other way, so one surface is not a shifted copy of the other.

**Where the loss is.** Concentrated in the biggest page (`layer_2`, 525k of the 1.07M common label pixels): 0.675 vs 0.775 for the official surface. The second page loses 0.03; the third (23 % ink) scores below chance for both.

**Depth scan** (stack shifted along the normal, `layer_2`, 530k labels, 50 % ink; pipeline page / official surface):

| shift (voxels) | -9 | -6 | -3 | 0 | +3 | +6 | +9 |
|---|---|---|---|---|---|---|---|
| pipeline page AUC | 0.594 | 0.683 | 0.468 | 0.674 | 0.489 | 0.441 | 0.503 |
| official surface AUC | 0.591 | 0.691 | 0.469 | 0.775 | 0.557 | 0.615 | 0.628 |

AUC swings between 0.44 and 0.78 within +-9 voxels (+-22 um), non-monotonically, with a rough 6-voxel zigzag and values below chance at -3 and +3, on both surfaces. **A signal that flips from 0.78 to 0.47 across 3 voxels (7 um) is not a robust ink detector** - it may be responding to the papyrus's layered
structure or to something that varies with depth in step with the label regions; this was not tested, and it is the most important finding here, more than the 0.08 gap. The two surfaces coincide at -9, -6 and -3 (within 0.01) and separate at 0 and beyond (by 0.07-0.17), and the pipeline's best (0.683) is below the official best (0.775).

**Re-refining the pipeline page** (same 530k labels; official surface 0.775):

| variant | AUC |
|---|---|
| as scored (M3 page, refined once, search +-8) | 0.674 |
| refined a second time (+-4) | 0.680 |
| refined again with finer smoothing | 0.647 |
| raw M3 page, one refine, search **+-12** | **0.719** |
| raw M3 page, search +-16 / +-20 | 0.747 / 0.706 |
| raw M3 page, search +-12 with heavier smoothing (6 / 2.5) | 0.646 |

Replication on the other labelled pages (as scored / +-12 / +-16 / +-20; official surface in brackets): `layer_13` 0.608 / 0.622 / 0.586 / 0.563 (0.644); `layer_1` 0.372 / 0.495 / 0.420 / 0.485 (0.356, below chance because only 23 % of its labels are ink).

**Reading it.** The default refine (search +-8, settling on the brightest smoothed ridge along the normal) sometimes lands on the wrong ridge, and a second pass cannot undo that. A search of +-12 improved all three labelled pages (+0.045, +0.014, +0.123) - the only setting that did - but the effect is small, the response to the search range is
erratic (+-16 helps one page and hurts another), and with about a dozen settings tried on one tile and an AUC that moves by ~0.1 with a few voxels of depth, this is as likely to be fitting depth noise as finding the right setting. **It is not enough to change the pipeline's default**; the gap to the official
surface is only partly explained and the rest was not located.

**Next step that would settle it:** the other four labelled Paris 4 segments (`w01`, `w02`, `w03`, `w05`; 205-430 MB of labels and geometry each, plus a ~1.5 GB CT crop per tile) would show whether +-12 holds and whether the depth zigzag is a property of the model or of this tile.

## Five segments, spatial-shift null and edge-band test (added 2026-10-23)

All five labelled Paris 4 segments (`w00`, `w01`, `w02`, `w03`, `w05`) have now been run through the same pipeline
(`scripts/probes/ink_segment_experiment.py`; tables from `scripts/probes/summarize_multi_segment.py`). The tables are: AUC per segment, the
gold-geometry depth scan (voxels along the normal), and the pooled spatial-shift null (`scripts/probes/ink_spatial_null.py`, JSONs in
`results/paris4_spatial_null/`).

| segment | tile labels | ink | share reached by pipeline (+-8) / official | common pixels | all-ink F0.5 | official+refine AUC | pipeline +-8 AUC | pipeline +-12 AUC |
|---|---|---|---|---|---|---|---|---|
| `w00` | 3.01M | 29% | 37% / 61% | 1070k | 0.457 | 0.729 | 0.652 | n/a |
| `w01` | 2.31M | 30% | 31% / 65% | 595k | 0.435 | 0.441 | 0.452 | 0.476 |
| `w02` | 1.43M | 47% | 15% / 94% | 183k | 0.415 | 0.533 | 0.371 | 0.441 |
| `w03` | 1.81M | 47% | 33% / 93% | 556k | 0.499 | 0.585 | 0.540 | 0.559 |
| `w05` | 2.47M | 51% | 16% / 92% | 336k | 0.519 | 0.657 | 0.553 | 0.586 |

| segment | -6 | -3 | 0 | +3 | +6 | page, labels (ink) |
|---|---|---|---|---|---|---|
| `w00` | 0.691 | 0.469 | 0.775 | 0.557 | 0.615 | layer_2, 530k (50%) |
| `w01` | 0.880 | 0.658 | 0.400 | 0.513 | 0.275 | layer_1, 500k (41%) |
| `w02` | 0.112 | 0.865 | 0.571 | 0.518 | 0.248 | layer_19, 77k (25%) |
| `w03` | 0.492 | 0.494 | 0.576 | 0.350 | 0.654 | layer_8, 457k (45%) |
| `w05` | 0.352 | 0.383 | 0.665 | 0.693 | 0.723 | layer_8, 160k (29%) |

| segment | ink map scored | observed AUC | null mean | null SD | null 5-95% | null range | shifts | two-sided p |
|---|---|---|---|---|---|---|---|---|
| `w00` | split pages | 0.653 | 0.483 | 0.165 | 0.23-0.79 | 0.08-0.98 | 249 | 0.36 |
| `w00` | baseline pages | 0.655 | 0.483 | 0.165 | 0.23-0.79 | 0.08-0.98 | 249 | 0.35 |
| `w01` | split pages, refine +-8 | 0.455 | 0.509 | 0.095 | 0.37-0.66 | 0.33-0.79 | 56 | 0.60 |
| `w02` | split pages, refine +-8 | 0.362 | 0.466 | 0.207 | 0.12-0.78 | 0.10-0.87 | 14 | 0.47 |
| `w03` | split pages, refine +-8 | 0.542 | 0.471 | 0.175 | 0.12-0.74 | 0.02-0.81 | 72 | 0.84 |
| `w05` | split pages, refine +-8 | 0.524 | 0.484 | 0.107 | 0.33-0.61 | 0.20-0.76 | 46 | 0.87 |

**Pooled AUC: chance on all five segments** (two-sided p 0.35-0.87). The depth-scan swings (for example `w02` 0.11 to 0.87 within +-3 voxels)
are inside the null range too.

**Edge-band test** (`scripts/probes/ink_edge_band.py`; JSONs in `results/paris4_edge_band/`; table from `scripts/probes/summarize_edge_band.py`).
Pooled AUC is decided by a few large blobs. Near a label edge, ink and no-ink pixels sit side by side, so slow trends (sheet brightness,
damage, depth drift) cancel out. Only points within B canvas pixels of an ink/no-ink edge are scored. The "high-passed" columns also
subtract the score map's local mean (Gaussian sigma 48 px). The null slides the whole label image, edge band included, by >= 400 px
(200 shifts; fewer on `w02`, whose window is small).

| segment | B (px) | points | band AUC | null 5-95% | p (2-sided) | high-passed AUC | null 5-95% | p (2-sided) |
|---|---|---|---|---|---|---|---|---|
| `w00_baseline` | 16 | 94,653 | 0.518 | 0.37-0.65 | 0.71 | 0.490 | 0.40-0.59 | 0.82 |
| `w00_baseline` | 32 | 195,737 | 0.548 | 0.27-0.70 | 0.52 | 0.490 | 0.34-0.68 | 0.88 |
| `w00_baseline` | 64 | 417,868 | 0.594 | 0.21-0.81 | 0.43 | 0.490 | 0.33-0.75 | 0.88 |
| `w00_split` | 16 | 95,159 | 0.516 | 0.37-0.65 | 0.73 | 0.492 | 0.40-0.60 | 0.84 |
| `w00_split` | 32 | 196,609 | 0.545 | 0.27-0.71 | 0.53 | 0.491 | 0.34-0.67 | 0.90 |
| `w00_split` | 64 | 419,428 | 0.589 | 0.21-0.80 | 0.42 | 0.485 | 0.33-0.75 | 0.83 |
| `w01` | 16 | 43,271 | 0.520 | 0.42-0.58 | 0.63 | 0.526 | 0.43-0.60 | 0.57 |
| `w01` | 32 | 89,105 | 0.514 | 0.37-0.64 | 0.80 | 0.546 | 0.39-0.62 | 0.46 |
| `w01` | 64 | 181,364 | 0.507 | 0.34-0.68 | 0.89 | 0.496 | 0.36-0.64 | 0.95 |
| `w02` | 16 | 22,866 | 0.478 | 0.29-0.64 | 0.79 | 0.471 | 0.33-0.66 | 0.71 |
| `w02` | 32 | 45,725 | 0.517 | 0.25-0.74 | 0.91 | 0.491 | 0.27-0.82 | 0.92 |
| `w02` | 64 | 87,522 | 0.514 | 0.16-0.70 | 0.94 | 0.477 | 0.26-0.79 | 0.78 |
| `w03` | 16 | 61,894 | 0.561 | 0.40-0.57 | 0.20 | 0.503 | 0.41-0.59 | 0.97 |
| `w03` | 32 | 125,723 | 0.574 | 0.34-0.66 | 0.30 | 0.495 | 0.38-0.63 | 0.93 |
| `w03` | 64 | 247,666 | 0.563 | 0.30-0.72 | 0.51 | 0.498 | 0.35-0.65 | 0.96 |
| `w05` | 16 | 26,311 | 0.551 | 0.42-0.58 | 0.27 | 0.487 | 0.43-0.57 | 0.73 |
| `w05` | 32 | 58,113 | 0.578 | 0.38-0.62 | 0.29 | 0.540 | 0.40-0.61 | 0.50 |
| `w05` | 64 | 132,313 | 0.589 | 0.33-0.65 | 0.39 | 0.597 | 0.41-0.63 | 0.18 |

**Result: no segment shows edge-following ink signal.** The lowest two-sided p is 0.18 (`w05`, B=64, high-passed). High-passed AUCs at
B=16 are 0.47-0.53. At B=16 the edge-band null is much tighter than the pooled one (`w01`: 0.42-0.58 vs 0.37-0.66), so this test has more
power. Even so, it can only detect an edge-band AUC above about 0.58-0.65 (the null's 95th percentile). A weaker real signal would be missed.
Raw band AUCs lean above 0.5 (4 of 5 segments at B=16), but a sign test gives p 0.38, so this is not significant.

**Conclusion:** the fragment-trained U-Net shows no detectable ink signal on Scroll 1 under either the pooled or the edge-band test.

## Limits

* The first analysis used one segment and one 1.5 x 1.4 x 0.8-mm tile (all five segments are covered in the section above); AUC is over spatially coarse labels, so spatial autocorrelation inflates confidence and a model that tracks "is this part of the sheet intact" could score above chance without detecting ink. No confidence intervals were computed at the time; the spatial-shift and edge-band nulls (section above) now serve as the reference distribution.
* Coverage is limited by the labels' ~8-voxel misalignment: at the default 8-voxel gate only 37 % of supervised pixels are scored, and the choice of gate moves AUC from 0.66 to 0.62 - larger gates admit more misaligned (and neighbouring-wrap) pixels.
* The model was trained at 3.24 um / 54 keV and applied here after resampling; no tuning to this scan was done.
* ~~This confirms the model carries some signal on a scroll with labels.~~ **Retracted (2026-10-12):** the spatial-shift null shows these AUCs are chance-level, so this test does **not** show the model carries ink signal on Scroll 1 (see the correction at the top and the edge-band test below). It also does **not** validate the PHerc. 0332 results (no labels there): the blob-shaped 0332 candidates remain unverified, and my earlier explanation of them as a response to smooth surface
  texture is speculation, not something this test supports or rules out.

## Reproduce

```bash
python scripts/eval_ink_labels.py --labels-dir data/real_scans/paris4/w00 --prefix w00_20231016151002 \
    --tile-origin 62080 17280 18816 --tile-shape 1536 1408 768 --gold data/real_scans/paris4/tile.h5 \
    --ink "baseline pages, normal depth=data/processed/paris4/baseline/normal/tile_ink.h5" \
          "split pages, normal depth=data/processed/paris4/split/normal/tile_ink.h5" --dmax 8
python scripts/render_label_comparison.py data/processed/paris4/split/normal/tile_ink.h5 --labels-dir data/real_scans/paris4/w00 \
    --prefix w00_20231016151002 --tile-origin 62080 17280 18816 --tile-shape 1536 1408 768 --out out.png
```
