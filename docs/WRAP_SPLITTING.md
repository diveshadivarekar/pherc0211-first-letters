# Splitting touching wraps (M2 `split_wraps`) and the page builder

## The problem

The surface prediction is thin, but where two wraps of the scroll touch it is one 3D-connected blob. M2 labels connected pieces
(cut into angular wedges), so a "segment" can hold several wraps - or sheets that overlap in a ray from the scroll axis. M3 then builds one page
from it, and the page cuts across or floats between the layers. The tifxyz QA ([TIFXYZ.md](TIFXYZ.md)) measured this on PHerc. 0332: before the
fix, 16.8 % of the mesh area (37.8 % on a second tile) was on pages spanning more than two wraps within one z-slice, and only 75-77 % of all
vertices lay on predicted surface.

## What was changed

`src/m2_segmentation/surface_segments.py::split_wraps`, on with `split_wraps_apart=True` (or a dict of options) in `label_surface_pieces` /
`segment_from_surface_prediction` and `--split-wraps` in `scripts/scan_tiles.py`. In cylindrical terms a sheet is a height field `r = f(theta, z)`:

1. Along each radial ray (a ~1-voxel arc by one slice) the occupied voxels split into **runs** at radial gaps - each run is one sheet crossing.
2. Runs in neighbouring columns **link** if their radial centres differ by at most `link_tol` (default 0.75 prediction voxel: a ramp steeper than about
   37 degrees is a bridge, not a sheet). Linked runs form a piece (union-find).
3. A run thicker than 1.6x the median is wraps in contact; it is divided radially into one sub-run per sheet, and a run thicker than 4 sheets is cut.
4. **Single-valued** (default): linked runs can still form a piece with two sheets in one ray (two sheets joined by a gentle path elsewhere), which no
   height field can represent. The runs of a piece in each column are ranked by radius and only equal-rank runs stay linked, so every piece has at most one
   run per ray - at the price of cutting a sheet where a neighbour starts or ends beside it (more, smaller pieces).

Splitting only splits pieces or drops junction voxels; it never merges. Vectorised numpy/scipy, a few seconds per tile.

M3 gained `sheet_unwrap` (`method="sheet"`, needs `--umbilicus`): a cylindrical height-field page builder. **It is opt-in; `auto` still uses the plane builder** - see below.

## Results (PHerc. 0332, 2.399 um; `scripts/eval_wrap_split.py`; raw numbers in [`tifxyz/wrap_split_results.json`](tifxyz/wrap_split_results.json))

Every row: M2 from the surface prediction -> M3 -> tifxyz stride 4 -> `scripts/tifxyz_qa.py`. "Multi-wrap area" = share of mesh vertices on pages spanning more than 2 wraps per
slice. Link tolerance in native voxels (prediction scale 4, so 3 = 0.75 voxel). Grid shear = 95th percentile of |cos| of the cell angle (0 = orthogonal grid).

**Tuned tile** (768 x 1024 x 1024 at (16384, 7680, 7680); the tolerance was chosen here):

| pipeline | pages | pages >= 10k vtx | vertices | multi-wrap area | on predicted surface | cross-wrap cells | grid shear |
|---|---|---|---|---|---|---|---|
| plain M2 + plane | 50 | 14 | 382k | 16.8 % | 75.2 % | 9.3 % | 0.35 |
| split tol 3 + plane | 82 | 23 | 530k | 0.0 % | 99.5 % | 0.4 % | 0.14 |
| split tol 2 + plane | 98 | 14 | 447k | 0.0 % | 99.5 % | 0.4 % | 0.12 |
| split tol 3 (multi-valued) + plane | 63 | 22 | 492k | 4.7 % | 91.8 % | 2.9 % | 0.21 |
| split tol 3 + sheet | 82 | 19 | 481k | 0.0 % | 99.6 % | 0.1 % | 0.55 |

**Held-out tile** (768 x 1024 x 512 at (16384, 7168, 7168); x 7168-7680, disjoint from the tuned tile's x 7680-8704; nothing was tuned on it):

| pipeline | pages | pages >= 10k vtx | vertices | multi-wrap area | on predicted surface | cross-wrap cells | grid shear |
|---|---|---|---|---|---|---|---|
| plain M2 + plane | 26 | 4 | 144k | 37.8 % | 76.8 % | 11.5 % | 0.44 |
| split tol 3 + plane | 101 | 9 | 291k | 0.0 % | 99.5 % | 0.5 % | 0.11 |
| split tol 2 + plane | 62 | 9 | 264k | 0.0 % | 99.6 % | 0.1 % | 0.11 |
| split tol 3 (multi-valued) + plane | 50 | 7 | 210k | 0.0 % | 78.6 % | 6.7 % | 0.33 |
| split tol 3 + sheet | 101 | 8 | 258k | 0.0 % | 99.7 % | 0.1 % | 0.40 |

## What it means

* **The single-valued rule is what removes off-surface pages.** Same tolerance, plane builder: on predicted surface 91.8 % -> 99.5 % (tuned) and 78.6 % -> 99.5 %
  (held-out). Without it, pieces still hold two sheets in 38-98 % of the radial columns of the worst pages (0.1-0.3 % on the good ones), and the page floats
  between them.
* **It generalises:** the recommended recipe (tol 3, single-valued, plane builder) takes the held-out tile from 37.8 % multi-wrap area and 76.8 % on-surface to 0.0 % and 99.5 %,
  with twice the mesh (144k -> 292k vertices) and more large pages (4 -> 9).
* **Tolerance:** 4-5 native voxels leave ramp bridges between touching wraps; 3 or below removes them (the sweep was 5, 4, 3, 2, 1 on the tuned tile). With the single-valued rule tol 2 and 3
  are equally pure (99.5-99.6 %) but tol 2 keeps less mesh (-16 % tuned, -9 % held-out) and fewer large pages on the tuned tile (14 vs 23). tol 1 fragments real sheets (no page over 10k vertices).
  0.75 prediction voxel is a starting point; sweep `--tol` on another scroll or resolution.
* **The new cylindrical page builder is not an upgrade.** It leaves slightly fewer cross-wrap cells (0.1 % vs 0.4-0.5 %) but its grid is much more sheared (cos 0.55 / 0.40 vs 0.14 / 0.11) and it covers
  less (482k vs 531k; 259k vs 292k vertices). The shear is intrinsic to a `(theta, z)` grid on a sheet whose radius changes along both directions - constant-theta columns do not
  remove it (tested: 0.57) - and only a real isometric flattening would. Shear distorts letter shapes, which is what ink reading depends on.
* **A wrong guess, corrected:** the plane builder was first blamed (it fits one plane per piece). A plane height field handles a 150-degree arc to within 1.8 voxels; the floating pages
  were multi-sheet pieces, which neither builder can represent.
* **A metric bug, corrected:** an early "wraps crossed" measure used the radial spread of the whole page, which counts a sheet that leans along z as several wraps; it overstated the baseline
  (54.5 % of the area instead of 16.8 %). `radial_extent` now measures within z-slices.

## Stitching the wedge pieces: how far the page builders can go

M2 cuts every sheet into angular wedges (`wedge_degrees`, default 40) so that each piece is a patch a page builder can handle. Fewer cuts mean larger
pages. Because the single-valued split already makes each piece one sheet, the wedge size is now just a parameter, and the question is how large a piece
the page builders can follow. Sweep on the tuned tile (single-valued split, tol 3; `scripts/eval_wrap_split.py --wedge N`; raw numbers in
[`tifxyz/wedge_size_results.json`](tifxyz/wedge_size_results.json)):

| wedge | page builder | pages | vertices | largest page | pages >= 20k | on predicted surface | cross-wrap cells | grid shear |
|---|---|---|---|---|---|---|---|---|
| 40 deg | plane | 82 | 530k | 20k | 3 | 99.5 % | 0.4 % | 0.14 |
| 80 deg | plane | 65 | 492k | 35k | 5 | 99.3 % | 0.4 % | 0.22 |
| 120 deg | plane | 56 | 456k | 42k | 6 | 98.9 % | 0.4 % | 0.33 |
| 180 deg | plane | 52 | 422k | 42k | 6 | 97.1 % | 1.6 % | 0.39 |
| 360 deg | plane | 45 | 414k | 43k | 8 | 95.7 % | 2.8 % | 0.42 |
| 180 deg | sheet | 52 | 473k | 56k | 6 | 99.5 % | 0.2 % | 0.60 |
| 360 deg | sheet | 45 | 469k | 57k | 7 | 99.5 % | 0.3 % | 0.62 |

* **Plane builder: good up to about 80 degrees.** Pages grow (largest 21k -> 35k vertices) with on-surface 0.995 -> 0.993; shear rises 0.14 -> 0.22 and total mesh drops 7 %.
  Past ~120 degrees it degrades (on-surface 0.989 at 120, 0.971 at 180, 0.957 for a whole turn; cross-wrap 0.4 % -> 2.8 %).
* **Cylindrical builder follows whole wraps** (360 degrees: on-surface 0.995, largest page 57.5k vertices, 2.8x the 40-degree plane page) **but the grid is sheared** (0.60-0.62
  vs 0.14-0.22). Stitching long sheets is therefore possible, at a distortion cost that matters for reading letters. Removing it needs a real isometric flattening
  (LSCM / ARAP on the mesh); the current `arap_relax` only rescales the grid and does not change page or coordinates.
* **Held-out tile (wedge 40 -> 80, plane builder):** largest page 21k -> 28k vertices, pages >= 20k 1 -> 2, pages >= 10k 9 -> 9, vertices 292k -> 285k, on-surface 0.995 -> 0.993,
  shear 0.11 -> 0.13. The gain is smaller than on the tuned tile and is not free.
* **Recommendation:** keep 40 degrees as the default; use `--wedge-degrees 80` (`scripts/scan_tiles.py`) when larger pages matter more than the last bit of mesh and grid regularity.
  No claim is made that larger pages read better - ink detection has not been re-run on them.

### A metric that does not survive large wedges

The sweep's first reading was alarming: "multi-wrap area" jumped from 0 % to 29 % at 120 degrees and to 35-51 % at a full turn. It is an artefact. The measure (median per-z-slab radial spread / wrap spacing) correlates 0.78
with the page's angular span, because the scroll is oval and one wrap's distance from the axis changes along its length. Pages of 100-150 degrees read as 1.8 wraps (median) while lying 98 % on predicted surface, and
the cylindrical builder's pages, which follow the labels exactly (on-surface 0.995), read as 41-51 % multi-wrap. `radial_extent` now reports `angular_span_deg` and `wraps_crossed_reliable` (span <= 90 degrees), and
the QA summary and the evaluation script count the measure only for reliable pages. For longer pages use on-surface and cross-wrap cells.

## Ink detection on the split pages, and the merged scroll

The fragment-trained U-Net (`ink_unet_20k.pt`, held-out fragment F0.5 0.52) was re-run on the 82 split pages of the tuned tile (tol 3, single-valued, plane builder, wedge 40) exactly as
`scripts/scan_tiles.py` runs it on the baseline's 50 pages: pages refined onto the papyrus (air pixels dropped), sampled along the normal, both depth orientations, candidates by
`find_candidates`. Raw numbers: [`tifxyz/ink_split_vs_baseline.json`](tifxyz/ink_split_vs_baseline.json).

| | baseline (plain M2) | split (tol 3, single-valued) |
|---|---|---|
| pages | 50 | 82 |
| page pixels -> read after refine | 6.12M -> 4.13M (32.5 % dropped as air) | 8.49M -> 6.21M (26.9 % dropped) |
| pixels with ink probability > 0.5, normal / flipped depth | 0.81 % / 0.85 % | 2.51 % / 0.77 % |
| candidate regions, normal / flipped | 1 / 0 | 1 / 2 |

What this does and does not show:

* **More of the scroll is read, with less thrown away.** 52 % more page area reaches the model and a smaller share is dropped as air - the floating-page problem the splitting removes, which the pipeline's refine step used to patch downstream.
* **There is no evidence of readable text, in either run.** The split run's one large candidate is a solid 276 x 231-voxel patch (36.8k pixels, mean probability 0.88, elongation 1.4, solidity 0.58 - strokes are long and thin)
  that appears in one depth orientation only (0.65 % of the merged page above 0.5 flipped, 9.9 % normal). The CT inside it has the same mean as the rest of its page but is smoother (std 0.101 vs 0.128); what drives it was not established (see [INK_EVALUATION.md](INK_EVALUATION.md): the labelled Scroll 1 test that once seemed to show some signal was retracted - its AUCs are inside a spatial-shift null - so these
  regions are unverified and there is no evidence the model detects ink on this scan). The baseline's single candidate (1.3k px) is a different place: **0 of 1 candidates in either run has a counterpart in the other run** (centres within 40 voxels),
  and 0 of 1 split candidates survives flipping the depth. A real stroke should be stable under a change of flattening.
* **Why that is expected.** The model was trained on 3.24 um / 54 keV fragments and applied to a 2.4 um / 78 keV scroll with no ink ground truth; its fragment F0.5 of 0.52 says nothing about this scroll. These results neither confirm nor refute ink on
  PHerc. 0332; cleaner pages do not make the model valid. That labelled test has since been done on Scroll 1 (PHerc. Paris 4) - see [INK_EVALUATION.md](INK_EVALUATION.md) - but not on PHerc. 0332, which has no labels.

### Merged pages (`scripts/merge_wedge_pages.py`)

Wedge pages are joined into one sheet when their pixels meet across a wedge seam at the same height and radius (touching different wraps sit at different radii). On the split pages: 61 seam joins turn 82 pages into **21 groups, 72 of the 82 pages in a
multi-page group**; the largest groups chain 16 pages over 169 degrees, 17 pages over 355 degrees (a whole turn at radius ~250), 10 pages over 258 degrees and 7 over 232. For comparison the baseline's 50 pages make 10 groups (largest: 21 pages over 249 degrees) - but those pages
each cross several wraps, so their groups are not clean single sheets. The merged page is a cylindrical projection about the scroll axis (angle x z at ~1 voxel per pixel at the group's radius), a view of the stitched sheet and not an isometric flattening.

* [`figures/merged_page_split_ink.png`](figures/merged_page_split_ink.png) - the largest split group (16 pages, 169 degrees): CT left, ink overlay right, blue ticks mark stitched wedge seams. The seams are not visible in the CT.
* [`figures/merged_overview_split.png`](figures/merged_overview_split.png) / [`figures/merged_overview_baseline.png`](figures/merged_overview_baseline.png) - every merged group on the (angle, z) map.
* [`figures/merged_page_baseline_ink.png`](figures/merged_page_baseline_ink.png) - the same for the baseline's largest group.

The seam rule (height and radius within 5 voxels, at least 30 facing pixel pairs) was not validated independently; joins were checked only by eye on the CT.

## Limits - what this does not show

* Purity is judged with this project's own QA (radial spread per slice, normal-vs-radial, support from the same prediction the segments were cut from) on two tiles of one scroll. It does
  **not** show the pages read better: there is no ink ground truth for this scroll, and re-running the ink model on the split pages (above) produced no verifiable text.
* On-surface is measured against the prediction the segments came from, so some of the gain is partly by construction; cross-wrap and multi-wrap area use the scroll axis, not the prediction.
* The single-valued rule makes smaller pages (median 4.6k -> 3.4k vertices on the tuned tile), because sheets are cut where neighbours start or end. Larger wedges recover some of that (see above) but
  a real long-sheet stitch needs an isometric flattening, which does not exist here yet.
* The tolerance default is relative to the prediction voxel and tuned on a scale-4 prediction. Rasterised toy rings jitter by ~1 voxel and fragment under it, so the unit tests pass a looser tolerance.

## Reproduce

```bash
python scripts/eval_wrap_split.py --name tuned   --origin 128 512 512 --shape 768 1024 1024 --tol 3 2 --ablate
python scripts/eval_wrap_split.py --name heldout --origin 128 0 0     --shape 768 1024 512  --tol 3 2 --ablate --min-voxels 50000
```

(needs `data/real_scans/pherc0332_native_crop.h5`, `pherc0332_surface_pred_crop.h5` and `pherc0332_umbilicus.json`; see `data/real_scans/README.md`).
