# PHerc. 0211 First Letters attempt: reproduction, controls and a documented negative result (Progress Prize entry, October 2026)

**Progress Prizes submission, October 2026 (form deadline 31 Oct 2026, 11:59 pm PT).**

* Team: Divesh Adivarekar (team leader, Discord: Divesh Adivarekar), Pranav Gugale, Rohan Gorde, Samarth Sangolgi
* Repository: https://github.com/diveshadivarekar/pherc0211-first-letters
* Licence: MIT (`LICENSE`)
* Date: 29 Oct 2026

## One-paragraph summary

We ran the published PHerc. 343 First Letters pipeline (villa spiral fit, snapping, VC3D render, `ink_9um` / Hecate / dense-native / Reader v2 ensemble) on
PHerc. 0211 (scan `20250821151803`, 9.362 um, 113 keV) across five z-bands, plus the Reader v2 model on 17 surfaces, with a positive control on the PHerc. 343
winner's own mesh in every stage. **We found no ink and no letters, and we make no letter claims.** The contribution is a reproducible, controlled negative
result with the failure modes and fixes we hit, an independent validation of our ink-label evaluation code against the official Paris 4 3D ink model, and a
corrected statement about what our earlier label-based tests can and cannot show. An independent run of the official workflow on another 0211 window
(bnleft/first-light-pherc0211) reports the same negative.

## Results at a glance

| # | Experiment | Result | Where |
|---|---|---|---|
| 1 | Positive control, PHerc. 343 winner mesh, reading side | `ink_9um` seed 42 AUC 0.59-0.60; 4-model ensemble 0.60-0.65 (peak 0.647 at d=+1) vs 0.40-0.51 on the blind side; shifted-box nulls up to ~0.61 | `docs/PHERC0211_SPIRAL_SCREEN.md` |
| 2 | 0211 spiral-fit screens, `ink_9um` seed 42 (level 1), bands z 7700-9200 (62 windings) and z 4700-6200 (66) | 0 flagged; reverse/forward coarse-std ratio 0.81-1.23 vs control 1.5-2.3 | same; `results/pherc0211_spiral_screen/` |
| 3 | 0211 4-model ensemble at level 0 (bands 1-5, selected windings, d=0) | no control-like asymmetry (control coherence 0.300/0.367 fw/rv, 0211 within +-0.03 and not systematically higher on one side) | same |
| 4 | Meshes snapped onto a sheet (our stand-in for the unpublished `snap_meshes.py`), 13 windings in 3 bands | same negative; the snap keeps the 343 control signal (AUC 0.623-0.634 vs 0.640-0.647 unsnapped) | same, "Snapped meshes" |
| 5 | Reader v2 (checkpoint `654ec5ac...d3a6`, MIT), 17 surfaces (13 own meshes, 4 community meshes) | 0 flagged (max 0.0064, flag 0.008); control 343 passes (0.021, author's text band 0.010-0.032) | `docs/PHERC0211_READER_V2.md` |
| 6 | Two leads followed up | `z7920_w040` +10..+13 voxel off-mesh response: rectangular blocks beside a damaged zone, not rows (also after sheet tracking). Band-1 `w050` hotspot: one isotropic ~2 x 1 mm blob cluster, no rows, neighbours flat, the opposite asymmetry on `w055` | READER_V2 doc; SPIRAL_SCREEN doc, "w050 hotspot follow-up" |
| 7 | Mesh QA: density-profile check | 600-3000-step fitted meshes sit at random depths relative to sheets (excess 0.04-0.11 vs 0.33 for the winner's snapped control); explains why snapping was added | SPIRAL_SCREEN doc |
| 8 | **Official Paris 4 3D ink model on our label code** (`scrollprize/ink_3d_dino_guided`, MIT, EMA weights, sha256 `5a148c2c...b303`) | pooled 9x9x9-max AUC 0.88 / 0.725 / 0.711 / 0.705 / 0.84 on w00 / w01 / w02 / w03 / w05, all outside the shifted-label null (p <= 0.03). In-sample (Paris 4 labels were in this checkpoint's training), so this validates our label/coordinate/null code, not generalisation | `docs/LITERATURE_GAP_ANALYSIS.md`, `results/ink3d_official/` |
| 9 | Edge-band test (ink vs no-ink pixels next to a label edge) | at chance even for the official model on most bands, so it cannot detect a working model on these labels. **Our earlier "edge-band: no signal on Paris 4" is therefore not evidence of absence** (the chance results of our own fragment U-Net on Paris 4 stand as a statement about that U-Net) | same; `docs/INK_EVALUATION.md` banner |
| 10 | Earlier feasibility work (0846A 2.4 um windows, 0211 small pages, wrap-splitting, tifxyz export + QA) | negative / methodology | `docs/PROGRESS_PRIZE_SUBMISSION.md`, `docs/WRAP_SPLITTING.md`, `docs/TIFXYZ.md` |

No result here is a claim of ink or text in PHerc. 0211. Rows 2-6 are label-free screens whose only anchor is the 343 control, so a negative means "no
control-like signal on these surfaces with these models", not "no ink".

## What is new / useful to others

* A **label-free control-anchored screening recipe** (reverse-vs-forward coarse-std, structure-tensor coherence, inter-model agreement, p99 z) that separates the 343
  control from 0211 and works at level 1 for cheap triage (`scripts/spiral0211/screen_L1.py`, `ensemble_*.py`).
* A **sheet snapping stand-in** (`scripts/spiral0211/snap_render.py`) with a test that it preserves the control signal.
* A **density-profile mesh QA** (`scripts/spiral0211/density_mesh_qa.py`) showing short spiral fits are off-sheet.
* An **evaluation harness for the Paris 4 ink labels** (`scripts/eval_ink_labels.py`, `scripts/probes/ink_spatial_null.py`, `ink3d_official.py`) with a spatial-shift null:
  the labels are a few large blobs, so naive pooled AUC on them is uninformative (nulls span 0.12-0.98 for a blobby model); a significant result needs the null.
  Scripts to run the official 3D model on local crops: `scripts/probes/ink3d_official.py`, `ink3d_official_all.sh`.
* Reader v3: a single resumable driver with a control gate and per-surface provenance (`scripts/reader_v3/run_reader_v3.py`; smoke-tested on the control only, no new 0211 results).
* Pitfalls (Windows commit-memory hangs in `vc_render_tifxyz`, Hecate is zero on CT holes so 4-model masks keep only 8-12 % of 0211, empty-strip pages that look like
  "no ink, controls pass", `/tmp` path mismatch, and others) are listed in the docs.

## Limitations (read these)

* Five of about twelve 1.5 cm z-bands of 0211 were screened; fits were 600-3000 steps (the winner used 30,000) and snapping is our own approximation.
* Models: `ink_9um` seed 42/43, dense-native, Hecate, Reader v2. All are trained on other scrolls; none has been shown to read 0211-type ink at 9.362 um. The scan may simply not resolve it.
* The Paris 4 validation is in-sample and its pooled AUC depends on a 9x9x9 max chosen after seeing w00 (fixed up front for the other four segments).
* No run was repeated with different random seeds for the screens; the Reader v2 / `ink_9um` inference is deterministic (a bit-identical re-run is documented).
* PHerc 343 control letters belong to the prize winner (Erwin Nieuwlaar); we use that mesh only as a positive control and do not present it as our result.

## Third-party material (not included in this repo)

| item | source | licence | note |
|---|---|---|---|
| CT volumes (scan `20250821151803` and others) | Vesuvius Challenge open data (S3 `vesuvius-challenge-open-data`) | CC BY-NC 4.0 | download with `scripts/download_*.py`; see `data/real_scans/README.md` |
| `ink_9um` seeds 42/43 | `scrollprize/ink_9um` | see model card | download by hand |
| Reader v2 `reader-v2-step040000.pth` | github.com/DomRusso2/reader-v2 | MIT | sha256 `654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6` |
| `ink_3d_dino_guided` `ckpt_78k_fullsup.pth` | huggingface.co/scrollprize/ink_3d_dino_guided | MIT | sha256 `5a148c2c1bb730bfa683f2b3e3cdfc2000424003605e42300b527ff90118b303` |
| Hecate, dense-native | `scrollprize/hecate`, `Nieuwlaar/ink9um-dense-native` | see model cards | |
| villa (`vesuvius`, `fit_spiral`, VC3D) | github.com/ScrollPrize/villa at commit `0fb38c45` (spiral chain), `a329895e` (ink inference) | see repo | |
| Winner's mesh/letter boxes (control) | github.com/Nieuwlaar/pherc343-first-letters | see repo | fetched as files, not redistributed here |
| Community 0211 atlas meshes | github.com/pscamillo/vesuvius-eligible-meshes @ `0c966f8f` | CC BY-NC 4.0 (data) | not redistributed; check prize rules before including |

## Reproduce

Environment used: Windows 11, RTX A5000 (24 GB); Python 3.14 + torch 2.11 (cu128) for the villa chain and ink models; Python 3.13 `.venv` (`requirements.txt`) for the
older pipeline and tests (`pytest`). Run one GPU job at a time on 24 GB. Commands are in the docs; the entry points are:

```bash
# 0211 spiral screen: tracks -> fit -> render -> screen -> ensemble (one band)
bash scripts/spiral0211/run_band.sh ...                       # see docs/PHERC0211_SPIRAL_SCREEN.md, "Reproduce"
python scripts/spiral0211/screen_L1.py ...                    # level-1 single-model screen
python scripts/spiral0211/snap_render.py ...                  # sheet snapping
python scripts/spiral0211/ensemble_run.py ...                 # 4-model ensemble
# Reader v2 / v3
python scripts/reader_v3/run_reader_v3.py <render.zarr> <name> --offsets 0 7 -7 --control <343 render> --control-name ctrl343 --control-offset 1
# Paris 4 validation of the official 3D model
python scripts/probes/ink3d_official.py infer data/real_scans/paris4/native_crop.h5 <ckpt_78k_fullsup.pth> data/processed/ink3d_official/w00_prob.npy
python scripts/probes/ink3d_official.py score data/processed/ink3d_official/w00_prob.npy data/real_scans/paris4/w00 w00_20231016151002 --out results/ink3d_official/w00.json
bash scripts/probes/ink3d_official_all.sh                     # w01-w05
pytest                                                        # unit tests for the older pipeline modules
```

`CHECKSUMS.sha256` covers every file in this bundle. The bundle is rebuilt from the main repo with `bash scripts/build_prize_bundle.sh`.
Seeds: no model was trained for this entry; `ink3d_official.py` uses numpy seed 0 for the label-shift null.

## Layout

```
README.md                 this file
docs/                     write-ups (SPIRAL_SCREEN, READER_V2/V3, INK_EVALUATION, LITERATURE_GAP_ANALYSIS, earlier feasibility draft) + figures/
results/                  small result tables: ink3d_official/, paris4_*/, pherc0211_spiral_screen/, pherc0211_hotspot/
scripts/                  spiral0211/, reader_v2/, reader_v3/, probes/, download + eval scripts
src/ tests/ configs/      older M1-M6 pipeline modules and their tests
data/real_scans/README.md download instructions (no data included)
```

## Honest framing for the reviewers

We are not asking for credit for letters. We submit (a) code and a documented procedure that others can reuse to screen a scroll cheaply and with a
positive control before spending days on fits, (b) the specific negative evidence for 0211 at 9.362 um so nobody has to repeat it, and (c) a correction to
our own earlier evaluation claims.
