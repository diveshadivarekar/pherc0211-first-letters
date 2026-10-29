# Progress Prize submission (draft): First Letters feasibility on PHerc. 0211 and 0846A, with a controlled ink-model null

**Status: earlier feasibility write-up (updated 2026-10-12); the current submission summary is the top-level `README.md`.**
Team: Divesh Adivarekar, Pranav Gugale, Rohan Gorde, Samarth Sangolgi - Repository: https://github.com/diveshadivarekar/pherc0211-first-letters (MIT) - Date: 2026-10-29

## Summary

We tried to reach the First Letters workflow on PHerc. 0211 (scan `20250821151803`, 9.362 um / 113 keV) and, after a survey of all 22 eligible
scrolls, on PHerc. 0846A's 2.403 um / 77 keV volume (two native windows, W0 and W1). **We did not find ink or letters, and we make no letter claims.**
What we contribute is (1) a reproducible, documented negative result with a shifted-surface control on two 2.4 um windows (our fragment model) and on
ten 9.4 um pages of 0211 with the community `ink_9um` checkpoint, where the model's output is not distinguishable from shifted copies of the same
surface, (2) a survey showing which
eligible scrolls have a 2.4 um volume at all, (3) a block survey whose density terms work at 19 um voxels but not at 38 um, with the check that showed it,
(4) failure modes and unit/format pitfalls we hit, and (5) small open tools: resumable bucket mirror, block survey and ranking, a plane-fit face
extractor, and a tifxyz export of the faces (VC3D-format meshes for human inspection).

## Findings (all numbers measured in this repo; commands below)

### 1. Only two eligible scrolls have a 2.4 um volume; their prize-listed IDs are the 9 um volumes
`scripts/survey_scrolls.py` (S3 listing, 2026-10-10): 9 scrolls have only a 9.362 um / 113 keV volume (0125, 0191, 0211, 0257, 0358, 0813, 0826, 0846B,
1545), 11 only 8.64 um / 116 keV (0175A/B, 0268, 0306B, 0343, 0483A/B, 0490A/B, 0800, 1218), and **PHerc0846A and PHerc1203 also have 2.403 um / 77 keV**
(`20260319102732`, `20260319130212`). The prize page lists the 9 um IDs (`20250728152254`, `20250820131727`) and is silent on whether the 2.4 um volume
may be used for First Letters (the Grand Prize forbids higher-resolution-derived data; only the Paris 4 title prize says "any scan").
**Question for the organizers: may a First Letters submission for 0846A / 1203 be built on the 2.4 um volume?**

### 2. At 9.4 um the official surface prediction separates sheets, but the stock M2 stage merges them
PHerc. 0211, four native crops (256 x 640 x 1536 voxels, level 0). Connected components of the prediction put 49.3 M of ~51.4 M surface voxels (~96 %)
into one piece in crop A; the page flattened from it had distortion 0.46. With wrap splitting (`--split-wraps`) the same crop gives 44 pages,
distortion mostly <= 0.03 (B: 87, C: 116, D: 151 pages). The largest page of each crop covers 9.1 / 16.7 / 15.0 / 5.8 mm^2 (A-D; valid pixels x
(9.362 um)^2; the page *bounding boxes* are 14.9 / 23.1 / 18.6 / 9.3 mm^2 - an earlier version of this draft quoted the bounding boxes as "9-23 mm^2"),
i.e. 24-70x smaller than the 4 cm^2 (400 mm^2) area in which a First Letters entry must show 10 letters. (Size is not itself disqualifying: the
rules accept smaller meshes covering "a specific region". But 10 letters need roughly 1 cm^2 or more at typical letter size, so pages this small could
barely hold them even if ink were present; the blocker is the absence of ink signal, not area.) The 12 largest pages (figures in `docs/figures/pherc0211/`) show a crossing
papyrus-fibre grid and no strokes. **Two of crop A's three largest pages (3.2 mm^2 each) are not papyrus at all:** they are strips at the crop edge
(x ~ 1524-1532) where the CT volume is exactly zero under every page pixel; they only look like "pages" because the surface prediction is non-zero there.
Sorting pages by area without checking the volume under them selects such strips (finding 4b).

### 3. At 2.4 um (0846A) layers and fibres resolve well; block selection needs a density term, and it only works at 19 um voxels
* **W0** (z 4608-5632, y 15104-15616, x 12544-14080), picked on planarity alone at level 4 (38 um voxels), is a delaminated, debris-filled periphery:
  faces of 3.6-6.8 mm^2 with large voids, bright debris and fold artefacts; no strokes at five depth offsets (`docs/figures/pherc0846a/`).
* At **level 4** the density terms could not have flagged it: its block sat at the 80th percentile of fraction-bright and the median of std.
* At **level 3** (19 um voxels, 3,092 papyrus blocks of 128^3) they do: of the 8 blocks overlapping W0, 7 are below the 30th percentile of papyrus-band
  fraction and all 8 are above the 60th percentile of void fraction; the saturated-debris fraction flags only 2 of 8 (98th/99th percentile).
  Requiring body fraction >= 0.727 (the median) and bright fraction <= 0.006 (the 90th percentile) keeps 1,378 blocks (45 %). **This is one known-bad
  region checked after the fact; we have no known-good region to test sensitivity, and low body fraction may simply mean "near the boundary".**
* **W1** (survey candidate 0: z 10240-11264, y 13568-14080, x 20224-21760) is 82 % papyrus-band voxels, 15 % void, 0.6 % saturated debris. But at native
  scale its layers are thin and crumpled: of the four plane-fit faces, face 0 is a multi-sheet blob (flatness 0.29) and faces 0 and 1 sit +-10 voxels
  (the search-window edge) off the CT ridge; faces 2 and 3 (6.4 and 5.2 mm^2) sit 5 voxels off. Planarity measured at 19 um does not transfer to the faces.
  No strokes are visible (`docs/figures/pherc0846a_W1/`).

### 4. Controlled ink check: the fragment-trained U-Net output is not evidence of ink on 0846A
Model: this repo's U-Net trained on the three official 54 keV / 3.24 um fragments (`ink_unet_20k.pt`, held-out fragment F0.5 0.52). **The community
`ink_9um` checkpoint is run separately in finding 4b (on 0211 only); KLAVIS and Nieuwlaar checkpoints were not run.** There is no ground truth for this scroll. Each face was run
in both depth orientations together with copies of the same surface shifted +-30 voxels along its normal (`scripts/probes/ink_on_faces.py`,
`ink_stats.json` in each figures folder). Share of valid pixels with P(ink) > 0.5 (the community `ink_9um` model was later run on 0211, finding 4b;
it was **not** run on the 0846A faces):

**W0** (`docs/figures/pherc0846a/`):

| surface | normal | flipped | surface | normal | flipped |
|---|---|---|---|---|---|
| face 0 | 25.1 % | 3.5 % | face 3 | 30.2 % | 0.0 % |
| face 0 +30 / -30 | 7.6 % / 14.3 % | 0.0 % / 0.3 % | face 3 +30 / -30 | 0.0 % / 1.2 % | 0.0 % / 3.5 % |
| face 1 | 10.5 % | 43.2 % | | | |
| face 1 +30 / -30 | **26.5 %** / 0.0 % | 0.0 % / 0.0 % | | | |

**W1** (`docs/figures/pherc0846a_W1/`; ~1 M valid pixels per surface, so these controls are well sized):

| surface | normal | flipped |
|---|---|---|
| face 2 | 0.2 % | 9.0 % |
| face 2 +30 / -30 | 4.3 % / 7.7 % | 1.4 % / 0.0 % |
| face 3 | 0.0 % | 0.0 % |
| face 3 +30 / -30 | 4.2 % / 3.6 % | 6.5 % / 2.9 % |

* Not specific to the surface: on W0 face 1 the +30 control fires more than the face; on W1 the real surfaces barely fire in the normal orientation
  (0.2 %, 0.0 %) while their shifted copies fire at 3.6-7.7 %. The response changes sign between windows. (On W0 face 3 the controls are mostly void -
  only 144 k valid pixels at +30 - so they are weak controls.)
* Orientation-dependent: reversing the depth direction changes W0 face 3 from 30 % to 0 % and swaps which orientation fires on W0 face 1.
* Not stroke-like: the high-probability regions are a few saturated round blobs (~1 mm on W0 face 3, 0.09-0.29 mm^2 on W1 face 2 flipped;
  `ink_check_face3_normal_UNVALIDATED.png`, `ink_check_face2_flip_UNVALIDATED.png`), unrelated to the fibre texture; the same smooth-blob behaviour as
  our labelled Paris 4 test, where observed AUCs lay inside a spatial-shift null (`results/paris4_spatial_null/*.json`; the older `docs/INK_EVALUATION.md` carries a correction banner and its "modest signal" claim is retracted).

### 4b. The community `ink_9um` checkpoint on PHerc. 0211 (in-distribution scan type): same picture, surface not distinguishable from controls
Section 4 used our own 54 keV fragment model, which is out of distribution at 9 um. The 0211 volume (9.362 um / 113 keV) is the scan type the
community `ink_9um` model was trained for, so this is the more informative test. Model: `hybrid_3d2d-seed42/step-075000.pth` (sha256
`e635558ae6a1a807a7e5ec1e83adfd45bc3c0ac53883ea43f1d4e085d62a9cab`), villa `vesuvius` at commit `a329895e` (2026-10-08). Per the checkpoint's own
config its training scrolls are 0139, 0814, 1667 and Paris 4 (no 0211, no 0846A), so there is no train/predict overlap here.
Input is a 17-layer surface volume sampled along the page normal at one voxel (9.362 um) per layer from the native crop; inference is the
official `vesuvius.ink_detection.inference.infer` (overlap 0.5, Hann blend). Each page is also run **reversed in depth** and as **copies of the
same surface shifted +-6 and +-14 voxels along the normal** (8 controls: 4 shifts x 2 depth directions). Share of valid pixels with P(ink) > 0.5
(`scripts/probes/ink9um_summary.py`, `docs/figures/pherc0211_ink9um/summary.json`; the three largest pages per crop, 10 scored pages, 9.1 mm^2 down to 4.2 mm^2):

| crop | page | area mm^2 | surface, normal | surface, reversed | controls min / median / max |
|---|---|---|---|---|---|
| A | layer_1 | 9.1 | 0.267 | 0.352 | 0.107 / 0.289 / 0.542 |
| B | layer_1 | 16.7 | 0.013 | 0.029 | 0.010 / 0.044 / 0.107 |
| B | layer_2 | 16.1 | 0.022 | 0.086 | 0.010 / 0.032 / 0.074 |
| B | layer_3 | 15.7 | 0.174 | 0.128 | 0.073 / 0.158 / 0.304 |
| C | layer_1 | 15.0 | 0.217 | 0.455 | 0.070 / 0.272 / 0.537 |
| C | layer_2 | 12.9 | 0.014 | 0.000 | 0.000 / 0.023 / 0.098 |
| C | layer_3 | 12.1 | 0.025 | 0.013 | 0.007 / 0.039 / 0.057 |
| D | layer_1 | 5.8 | 0.081 | 0.062 | 0.051 / 0.063 / 0.163 |
| D | layer_2 | 4.2 | 0.185 | 0.211 | 0.087 / 0.400 / 0.545 |
| D | layer_3 | 5.1 | 0.125 | 0.079 | 0.129 / 0.257 / 0.495 |

(A layer_2 and layer_3 were also run: their input is all zeros, the volume being zero under them, so their all-zero output is not a result and they
are excluded.)

* **No specificity to the surface.** In 19 of the 20 (page, depth-direction) comparisons the surface value lies at or below the largest control; the one
  exception is B layer_2 reversed (0.086 vs 0.074, a 1.2-point margin on a page where at most 9 % of pixels fire at all). With 8 controls and no
  real signal one would expect the surface to be the largest in about 1 of 9 comparisons, i.e. ~2 of 20; we see 1. On 5 of the 10 pages (B layer_1,
  C layer_2, C layer_3, D layer_2, D layer_3) both depth directions are below the *median* control.
* **No strokes.** By eye the high-probability regions are smooth blobs roughly 0.5-1.5 mm across
  (`docs/figures/pherc0211_ink9um/<crop>/layer_*_ink9um_UNVALIDATED.png`), not thin row-aligned strokes. The page with the highest shares (C layer_1)
  renders as a bright featureless patch with masked holes and no visible fibre texture, so its high P(ink) is more likely a density artefact than text.
* **Deterministic:** re-running B layer_2 (normal) gives a bit-identical output TIFF (max abs difference 0 on a 269 x 759 uint8 image).
* **Not evidence of ink, and not evidence of absence** (see limitations): the controls are neighbouring layers of the same papyrus, so the test asks
  only whether the output *depends on the surface*. It does not.

### 5. The faces are exported as tifxyz so they can be inspected in VC3D (not submitted as meshes of text)
`scripts/export_tifxyz.py` writes each face as `x.tif / y.tif / z.tif / mask.tif / meta.json` in level-0 voxel coordinates of volume `20260319102732`
(`--offset` = the crop origin; stride 4, scale ~0.25). The built-in contract check (`check_contract`) reported no findings for any of them:

| mesh | grid | valid vertices | area |
|---|---|---|---|
| `pherc0846a_W1_face2` | 382 x 343 | 69,467 | 0.065 cm^2 |
| `pherc0846a_W1_face3` | 353 x 272 | 56,828 | 0.060 cm^2 |
| `pherc0846a_W0_face0` | 375 x 337 | 73,440 | 0.074 cm^2 |
| `pherc0846a_W0_face1` | 314 x 281 | 50,786 | 0.048 cm^2 |
| `pherc0846a_W0_face3` | 264 x 214 | 38,758 | 0.039 cm^2 |

(`area_cm2` from each `meta.json`: mesh surface area, which differs a little from the valid-pixel areas quoted above.) The largest is about 1/54
of the 4 cm^2 area in which 10 letters must appear (a smaller mesh is allowed, but these are too small to hold 10 letters at a typical letter size);
all are plane-fit height fields (distortion not measured) and contain no legible text. They exist so a reviewer can open the exact surfaces we looked at against the 2.403 um volume.

The ten scored 0211 pages of finding 4b are exported the same way (`meshes/pherc0211/pherc0211_<crop>_<layer>/`; stride 4, scale ~0.25, level-0
coordinates of volume `20250821151803` using the crop origins A (z9600 y3840 x4608), B (z8320 y4096 x1024), C (z6272 y4608 x2048), D (z11904 y1536
x4096); contract check: 0 findings for all 15 meshes in the package). Mesh areas (cm^2): A_1 0.092, B_1 0.165, B_2 0.160, B_3 0.156, C_1 0.145,
C_2 0.127, C_3 0.116, D_1 0.057, D_2 0.041, D_3 0.050. Each mesh's `meta.json` records its source page; the corresponding `ink_9um` outputs are in
`docs/figures/pherc0211_ink9um/`. These are page-level exports of our own flattening, **not** isometric flattenings measured for distortion, and we have
not opened them in VC3D.

## Pitfalls we hit (so you do not have to)
* `m7-L2` prediction crops on the 2.4 um volume are **4x coarser than native**; `download_surface_prediction.py` takes coordinates in *prediction* voxels
  (we first passed native coordinates and got "crop exceeds the prediction volume").
* 0846A has no umbilicus published (0211 and 0826 do), so M2's angular wedge splitting is unavailable; we used a plane-fit face extractor instead, which
  fails on crumpled sheets (W1 face 0).
* A "sheet period" from an FFT of a profile through 19-38 um voxels is clipped at the band edge and measures nothing; only planarity and the density
  fractions were usable, and only the latter at level 3 or finer.
* Block planarity at 19 um does not predict whether a native-resolution face can be extracted (W1).
* Per-pixel ridge snapping on these crops saturates its +-10 voxel window; do not trust snapped surfaces for fine detail.
* Native 2.4 um crops of 0.75 GiB took 3.5 min for W0 and 30 min for W1 on the same machine: bandwidth varies a lot, and the 38 GB level-1 mirror of 0211
  competing for the link did not help.
* **Page area ranking picks empty strips.** Two of crop A's three largest 0211 "pages" lie where the CT is exactly zero (crop edge), so a model run on
  them returns a perfectly flat 0 that looks like "no ink, controls pass". Check that the volume under a page is non-zero before scoring it
  (`ink9um_summary.py` now lists such pages as EMPTY instead of scoring them).
* A page's bounding box overstates its area by 1.2-1.6x (14.9 vs 9.1 mm^2 for A layer_1); report valid-pixel area.
* The `ink_9um` inference script wants a surface *volume* (depth x H x W zarr, here 17 layers), not a single flattened page; a page file from M3 has
  only one layer, so the stack has to be re-sampled from the native crop along the normal.
* In a Windows setup with both Git Bash and a native Python, `/tmp/...` paths are not the same file to the two: an output written by Python to
  `/tmp/x.tif` went to `D:\tmp`. Use relative paths.

## Reproduce
```bash
# 0211 data (levels 1-5 + umbilicus + surface-prediction levels 2-5), resumable
python scripts/download_s3_prefix.py --out data/real_scans/pherc0211 --strip PHerc0211/ --prefix <see data/real_scans/README.md>
python scripts/survey_scrolls.py PHerc0125 ... PHerc1545 --json data/processed/scroll_survey/s3_survey.json
# 0846A 2.4 um: levels 3-5 (9.2 GB), block survey at level 3, ranking with the density filter
python scripts/download_s3_prefix.py --out data/real_scans/hires --strip "" \
    --prefix PHerc0846A/volumes/20260319102732-2.403um-0.2m-77keV-masked.zarr/3 --prefix <same>/4 --prefix <same>/5 --prefix <same>/.zattrs
python scripts/survey_regions.py --zarr <0846A 2.4um zarr> --level 3 --voxel-um 2.403 --out PHerc0846A_L3.csv
python scripts/rank_regions.py --csv PHerc0846A_L3.csv --top 6 --min-sep 6000 --min-body 0.727 --max-bright 0.006 --depth 1024 --ysize 512 --xsize 1536
# windows W0 (z4608 y15104 x12544) and W1 (z10240 y13568 x20224); prediction crop coordinates are in prediction voxels (native / 4)
python scripts/download_native_crop.py --out native_crop_W1.h5 --scroll-id PHerc0846A --volume 2.403um --voxel-um 2.403 \
    --z0 10240 --y0 13568 --x0 20224 --depth 1024 --ysize 512 --xsize 1536
python scripts/download_surface_prediction.py --out surface_pred_crop_W1.h5 --scroll-id PHerc0846A --name-contains m7-L2 --scale 4 \
    --z0 2560 --y0 3392 --x0 5056 --depth 256 --ysize 128 --xsize 384
python scripts/probes/face_from_prediction.py --crop native_crop_W1.h5 --pred surface_pred_crop_W1.h5 --out out --top 4
python scripts/probes/ink_on_faces.py --crop native_crop_W1.h5 --pred surface_pred_crop_W1.h5 --model data/processed/models/ink_unet_20k.pt --faces 2 3 --out out_ink
python scripts/export_tifxyz.py <faces h5 of the unshifted layers> --out tifxyz/pherc0846a_W1 --offset 10240 13568 20224 --voxel-um 2.403 --stride 4
python scripts/pherc0211_tile.py A --split-wraps      # 0211 crops A-D via scripts/download_*.py as in data/real_scans/README.md
# finding 4b: community ink_9um on the three largest 0211 pages per crop with shifted / reversed controls (run with the env that has villa `vesuvius`)
sh scripts/probes/run_ink9um_0211.sh                  # ~45 min on an RTX A5000 (24 GB): 4 crops x 3 pages x 5 shifts x 2 directions
python scripts/probes/ink9um_summary.py data/processed/pherc0211_ink9um --flat-root data/processed/pherc0211_split --json docs/figures/pherc0211_ink9um/summary.json
python scripts/probes/ink9um_figures.py data/processed/pherc0211_ink9um/B --out docs/figures/pherc0211_ink9um/B   # likewise A, C, D
```
Environment for 4b: Python 3.14.4, torch 2.11.0+cu128, numpy 2.5.3, zarr 3.4.1, scipy 1.18.1, NVIDIA RTX A5000 24 GB; inference uses `--no-compile`.
The checkpoint is not redistributed here; its sha256 is given in finding 4b. We did not record the URL it was downloaded from.
Code commits on `main`: `6184491`, `638e5f7` (PRs #4, #5), `cee22cc`, `aafe238`, `eae1c40`, `9db375a`. Finding 4b's scripts (`scripts/probes/ink9um_on_pages.py`, `ink9um_summary.py`, `ink9um_figures.py`, `run_ink9um_0211.sh`) and `docs/figures/pherc0211_ink9um/` were committed in `9e95bba`; later work is in `98fd169`, `595f9de`, `e8f7386` and the 2026-10-23 limitations commit (the submitted commit is the latest on `main` of the submission repository). Seeds: no model was trained for this entry. Determinism: one `ink_9um` inference (B layer_2,
normal) was re-run and is bit-identical; the fragment-model runs of section 4 were not re-run. Software versions for the fragment-model runs
(`.venv`): Python 3.13.3, torch 2.9.0+cu130, numpy 1.26.4, scipy 1.17.1, scikit-image 0.26.0, h5py 3.16.0, zarr 2.18.7, tifffile 2026.3.3,
s3fs/fsspec 2026.9.0, numcodecs 0.15.1.

## Limitations (please read)

* **Update 2026-10-24.** Snapped meshes: re-screening 13 PHerc 0211 windings after snapping them onto a sheet (`scripts/spiral0211/snap_render.py`; the snap keeps the PHerc 343 control's signal) gives the same negative, apart from one local 2 x 1 mm blob cluster on band-1 winding 50 that is not text (`docs/PHERC0211_SPIRAL_SCREEN.md`, "Snapped meshes"). Scroll 1 (Paris 4): our own model is chance on all five labelled segments under the pooled spatial null and the edge-band test (`docs/INK_EVALUATION.md`). **Correction 2026-10-29:** the official `ink_3d_dino_guided` model, run through the same label code, scores 0.70-0.88 pooled on all five segments (in-sample), but is at chance on most edge-band tests, so the edge-band test cannot detect ink on these labels and our edge-band result is not evidence of absence (`docs/LITERATURE_GAP_ANALYSIS.md`).
* Two models only: ours (fragments at 54 keV; out of distribution on both scrolls) and one `ink_9um` checkpoint (`hybrid_3d2d-seed42`, step 75000).
  `ink_9um` was run on **0211 only**, not on the 0846A faces (2.4 um; it would need a 4-voxel step or a 4-layer mean to emulate 9.6 um layers, which we
  did not implement). Other community checkpoints (KLAVIS, Nieuwlaar, other `ink_9um` seeds) were not run.
* The model's 128 x 128 patch covers ~1.2 mm, about half the height of the 2.5 mm-tall pages scored; a large window can produce smooth plausible
  blobs without ink, which is what we see. A model that cannot tell the surface from its shifted copies tells us nothing about text *there*.
* Shifted-surface controls can land on another ink-bearing layer or in void; W0 face 3's controls were mostly void. For 0211 the sheet spacing is ~7-13
  voxels, so the +-6 controls sit between sheets and the +-14 controls near the next sheet; this tests dependence on the surface, not specificity to ink.
* The 0211 `ink_9um` run scored only ten pages of 4.2-16.7 mm^2 (the three largest per crop, A: one page + two empty strips), not the whole scroll; at
  9.4 um the visible fibre texture is coarse, and absence of ink output on these pages says nothing about other parts of 0211.
* Visual inspection covered 12 pages (0211) and 4 + 2 faces (0846A W0, W1) only; absence of visible ink here is not absence of ink in the scrolls.
* PHerc. 1203 (not worked on here, per the team's decision) has 22 organizer-grown auto-grown tifxyz segments in the open bucket
  (`PHerc1203/segments/raw/`, 2.9-16.1 cm^2 each by their `meta.json`), unlike 0211 and 0846A, which have none; we only read the listing and the
  metadata, we did not render them.
* The density filter was checked on one known-bad window only (see finding 3); its sensitivity to good regions is untested.
* Face areas assume a 1-voxel grid; faces are height fields over a plane, not isometric flattenings (distortion not measured), and the exported tifxyz
  meshes were checked with this repo's contract check, and VC3D's own renderer `vc_render_tifxyz` (2026-10-23) loads a 0211 split-wraps page (`pherc0211_A_layer_1`) and renders papyrus at its stated coordinates (`docs/figures/pherc0211/vc3d_render_A_layer_1.png`); we have **not** opened them in the VC3D GUI.
* Statements about third-party work (Miller & Muller PHerc 0826, Nieuwlaar ink9um-dense-native) were read from the public pages and READMEs, not reproduced.

## Data, licence, credits
Scroll data: Vesuvius Challenge open data, CC BY-NC 4.0. Our code: MIT (see `LICENSE`).
Prediction/surface models are the organizers' (`20260413222639`). The work was done with AI-tool assistance (disclosed, as in the
prior PHerc 0826 entry); every number above was produced by the commands listed.
