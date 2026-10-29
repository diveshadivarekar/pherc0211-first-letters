# PHerc. 0211 (scan 20250821151803): spiral-fit screen for ink, reproducing the PHerc. 343 First Letters pipeline

Status: **no ink signal found**, with the winner's 4-model ensemble validated on their own PHerc 343 mesh. This is an honest negative result for one z-band and crude 600-step meshes; it is not a proof that the scroll carries no ink.

## What was run

Method follows Nieuwlaar/pherc343-first-letters (PHerc. 343 First Letters winner): villa spiral fit on whole windings, 66-layer render along the
normal, `ink_9um` ink model, depth window of 17 layers centred on `d = layer - 33`.

| Step | 0211 setting |
|---|---|
| Band | z 7700-9200 (level-0 voxels); umbilicus drift ~100 vox over the band; ~65 windings, outer radius ~1900 vox |
| Inputs | m7 surface prediction (L0/L1), lasagna nx/ny (group 2, scale 4), umbilicus, 163,125 skeleton tracks, surface SDT built with `make_surf_sdt.py` |
| Fit | villa commit 0fb38c45 `fit_spiral.py`, CW, patches off, shell losses 0, `dense_spacing_mode=phase`, **600 steps** (not 30,000), `shell_outer_winding_idx=72` (guess) |
| Render | VC3D `vc_render_tifxyz`, 66 layers, 1 voxel apart (level 0 for 4 windings, level 1 for the screen) |
| Model | `ink_9um` hybrid_3d2d seed 42, step 75,000 (1 of the winner's 4 models) |

Deviations from the winner: `track_crossing_mode=count` (track_walk no longer exists in the native module), winner's `snap_meshes.py` + lasagna flatten
not run, 600-step fit instead of 30k, single model instead of a 4-model ensemble, local patches for Windows and the missing outer shell.
The CW/ACW choice could not be decided from loss: 600-step fits were indistinguishable (same loss, same on-surface fractions).

## Positive control (PHerc. 343, winner's own mesh)

Winner's `concat_w047-w048` mesh and letter boxes (their public repo), rendered with the same recipe, `ink_9um` seed 42:

| depth | reverse ("reading side") AUC, letter boxes vs rest | forward AUC | reverse coarse_std / band_std / coherence | forward |
|---|---|---|---|---|
| -4 | 0.592 | 0.538 | 28.9 / 5.6 / 0.345 | 18.3 / 4.3 / 0.282 |
| 0 | 0.598 | 0.462 | 30.7 / 5.9 / 0.359 | 18.6 / 4.1 / 0.277 |
| +1 | 0.600 | 0.444 | 30.8 / 5.9 / 0.358 | 19.4 / 4.1 / 0.276 |
| +4 | 0.558 | 0.436 | 27.2 / 5.5 / 0.331 | 20.4 / 4.3 / 0.286 |
| +8 | 0.501 | 0.461 | 16.5 / 3.8 / 0.257 | 19.0 / 4.0 / 0.267 |

Shifted-box null AUCs are 0.42-0.57, so a single model gives only a weak AUC; the label-free signature is stronger:
ink present means reverse-direction maps with much higher stroke-band energy and coherence than forward, peaking within +-4 of the surface.
The same signature survives at half resolution (level 1): reverse 29.7-30.5 / 5.8-6.0 / 0.38-0.39 vs forward 12.6-19.5 / 2.9-3.9 / 0.26-0.30.

## 0211 results

* Level-0 renders with depth sweeps: w045 (d -12..+12), w020 and w035 (d -8, 0, +8), w055 (d -8, 0, +8): forward ~ reverse, no depth peak,
  coarse_std 16-27, band_std 3.6-5.1, coherence 0.28-0.33. Maps are smooth 0.5-1 mm blobs.
* Level-1 screen at d=0, all 62 windings (first pass 13, 16, ..., 64: `data/spiral/screen_L1/summary_coarse18.txt`; full table via `screen_summary.py`): max reverse coarse_std 24.8
  (control: ~30), reverse/forward ratio 0.65-1.05 (control: 1.5-2.3), **0 of 62 flagged** by the rule
  `rv coarse_std >= 26 and rv coherence >= 0.33 and ratio >= 1.3`. Outer windings have higher texture in both directions (papyrus fibre structure), not asymmetric.

## 4-model ensemble check (winner's models: ink_9um seed 42 + 43, dense_native, Hecate 9.6 um)

Rule as in the winner's README: per-model z-score, ensemble = mean z; layers `25+d .. 41+d` (17) for the villa models, Hecate on 16 planes
resampled to 9.6 um; forward = blind side, reverse = reading side. Scripts: `ensemble_run.py`, `ensemble_score.py`, `ensemble_compare.py`.

PHerc 343 control (winner's mesh, letter boxes; AUC = letter boxes vs rest, reading side): seed 42 0.59-0.60, seed 43 0.55-0.62,
dense_native 0.57-0.60, **Hecate 0.67-0.72**, ensemble 0.60-0.65 (peak d=+1: 0.647); forward side 0.40-0.51; shifted-box nulls 0.42-0.61.
Inside the boxes the reverse-side ensemble z has p99 3.2; outside (>150 px from any box) 0.7-1.0.

Label-free comparison at d=0 (level-0 renders, 600-step CW meshes):

| render | coherence fw/rv | p99(z) fw/rv | model agreement fw/rv | corr(fw, rv) |
|---|---|---|---|---|
| 343 control (ink) | 0.300 / 0.367 | 2.28 / 2.96 | +0.26 / +0.66 | +0.14 |
| 0211 w020 | 0.342 / 0.337 | 2.18 / 2.35 | +0.40 / +0.29 | +0.12 |
| 0211 w035 | 0.315 / 0.316 | 2.24 / 2.92 | +0.29 / +0.39 | +0.13 |
| 0211 w045 | 0.321 / 0.325 | 2.31 / 2.58 | +0.45 / +0.47 | -0.05 |
| 0211 w055 | 0.286 / 0.282 | 2.78 / 2.28 | +0.41 / +0.21 | +0.23 |
| 0211 w065 | 0.285 / 0.276 | 2.62 / 2.56 | +0.27 / +0.31 | +0.13 |

The control shows one side (reverse) clearly different on all three measures. 0211 shows no consistent direction asymmetry in coherence
(differences within +-0.005), the side with heavier tails varies from winding to winding, and model agreement is not systematically
higher on either side. Ensemble maps of w045 (`data/spiral/preview/w045_ens_d0.png`) are scattered blob clusters that appear on both sides
at the same places, not rows of glyph-like blobs. Conclusion: no letter-like ink signal in the five windings tested with the 4-model ensemble.

## Second band: z 4700-6200

Same pipeline, new inputs (190,974 tracks; 3,000-step CW fit, track-radius loss ~145 vs ~880 for the 600-step band-1 fits;
`shell_outer_winding_idx=76`; windings 10-75; on-surface within 8 vox 89-92%).

* Level-1 single-model screen (ink_9um seed 42, d=0), all 66 windings: coarse_std median fw 27.3 / rv 26.6 (band 1: 22.5 / 20.5),
  rv/fw ratio 0.81-1.23 (control: 1.5-2.3), **0 asymmetric windings**. Texture is higher in both directions in this band.
  The level-1 maps contain large (2-3 mm), sharp-edged blobs on both sides, but at different places for forward and reverse.
* Level-0 4-model ensemble at d=0 on windings 25, 40, 55, 70:

| render | coherence fw/rv | p99(z) fw/rv | model agreement fw/rv | corr(fw, rv) |
|---|---|---|---|---|
| 343 control (ink) | 0.300 / 0.367 | 2.28 / 2.96 | +0.26 / +0.66 | +0.14 |
| 0211 b2 w025 | 0.305 / 0.321 | 2.13 / 2.28 | +0.21 / +0.38 | +0.25 |
| 0211 b2 w040 | 0.299 / 0.294 | 2.51 / 2.46 | +0.29 / +0.32 | +0.42 |
| 0211 b2 w055 | 0.290 / 0.293 | 2.58 / 2.14 | +0.33 / +0.23 | +0.35 |
| 0211 b2 w070 | 0.301 / 0.294 | 2.48 / 2.15 | +0.30 / +0.28 | +0.30 |

Band-1 windings 30 and 40 added: w030 0.298/0.317, 2.07/2.74, +0.22/+0.35 (weakly reverse-leaning, well below control); w040 0.318/0.307, 2.42/2.29, +0.32/+0.23.
Ensemble maps for w025 (b2) are scattered smooth blobs at L0 (no rows or glyph-like shapes); the larger level-1 blobs do not reproduce at level 0.
Conclusion for this band: no letter-like ink signal.

Caveat: the control signature is one-sided (reading side high, blind side low). A response that is equally strong on both sides would not be flagged by the
asymmetry rule; the level-0 ensemble tail statistics (p99 2.1-2.6, Gaussian ~2.33) and the maps argue against that here.

## Limits

Two 1.5 cm z-bands (z 7700-9200 and 4700-6200) of an ~18 cm scroll; single model; 600-step meshes; one depth in the screen; the signature thresholds come from one control;
the level-1 screen covered all 62 windings (0 flagged; median reverse coarse_std 21.1, max 24.8). Not tested: z-bands other than these two (about 11 of the scroll's 12 more 1.5 cm bands), a 30k-step fit, the ensemble at depths other than d=0 or on windings other than 20/35/45/55/65.

## Reproduce

`scripts/spiral0211/` (`run_tracks.py`, `run_fit.sh`, `render_windings.sh`, `ink_depth_sweep.sh`, `screen_L1.py`, `screen_summary.py`,
`control_343_score.py`, `ink_structure_by_depth.py`, shim `shim/vc`). Data under `data/spiral/` and `data/control_343/` (not committed).

## Density-profile mesh QA (`scripts/spiral0211/density_mesh_qa.py`)

Raw CT density does not mark ink even on the PHerc 343 control (letter boxes vs rest, 66 layers: AUC 0.46-0.54 raw, 0.49-0.50 high-passed;
fine-band texture 0.40-0.46 but shifted-box nulls give the same), so density is used for geometry QA instead. For each 96 px tile of a
66-layer render the mean density per layer is taken; the sheet is the prominent peak nearest the mesh layer. A mesh that lies on a sheet puts
that peak at one consistent offset; a mesh at random positions relative to the sheets spreads the offsets uniformly within +-(sheet spacing / 2).
`excess_concentration` = (share of tiles within +-3 layers of the modal offset - same share under random placement) / (1 - random share).

| render | holes % | strong-peak % | modal offset | on-sheet % | random % | excess |
|---|---|---|---|---|---|---|
| 343 control (winner's snapped mesh) | 9.3 | 89 | +3 | 49.4 | 24.6 | **0.33** |
| 0211 w020 | 16.7 | 97 | +4 | 27.7 | 23.2 | 0.06 |
| 0211 w030 | 16.7 | 90 | -7 | 27.1 | 22.9 | 0.05 |
| 0211 w035 | 16.7 | 84 | 0 | 31.8 | 24.2 | 0.10 |
| 0211 w040 | 16.7 | 77 | +4 | 30.9 | 22.7 | 0.11 |
| 0211 w045 | 16.7 | 78 | 0 | 29.5 | 24.6 | 0.06 |
| 0211 w050 | 16.7 | 68 | +5 | 27.1 | 23.8 | 0.04 |
| 0211 w055 | 16.7 | 66 | -1 | 29.3 | 23.3 | 0.08 |
| 0211 w065 | 16.7 | 55 | +1 | 20.9 | 23.8 | -0.04 |

Reading: the 600-step 0211 meshes sit at near-random positions relative to the sheets (excess 0.04-0.11 vs 0.33 for the control mesh), so the ink
models above were run on surfaces that are mostly not on a sheet. **The 0211 ink result is therefore weak evidence of absence, not a clean
negative**; a 30k-step fit plus snapping must reach an excess comparable to the control before the ink check is repeated (snapping done 2026-10-24, see "Snapped meshes" below). Holes are the same
16.7 % for all windings because the top and bottom strips of each render lie outside the band. The sheet spacing is 15-17 layers (140-160 um) in
both scrolls, so the 66-layer window always contains 2-4 sheets and a second density peak is normal, not a defect.
Limits: one control, tile-level (96 px) profiles, no degraded-control calibration yet.

## Snapped meshes (2026-10-24): re-screen after putting the meshes on a sheet

The density QA above showed that the 600-3000-step meshes sit at random depths relative to the sheets (excess 0.04-0.11 vs 0.33 for the
winner's snapped control). The winner's `snap_meshes.py` is not public, so `scripts/spiral0211/snap_render.py` is our stand-in. It resamples the
existing 66-layer render along the mesh normal so that the centre layer follows one sheet:

* Candidates are the CT depth-profile peaks within +-8 layers of the mesh (half a sheet spacing, so the snap cannot reach the next sheet).
* Region growing allows at most 2 layers of change per 4 px, strongest peak first, so the surface stays on one sheet.
* Pixels with no sheet within +-8 layers become holes.

`scripts/spiral0211/snap_screen.py` is the seed42 forward/reverse screen on the snapped stacks.

| render | assigned % of valid | patches | offset p5 / median / p95 (layers) | density excess before -> after |
|---|---|---|---|---|
| 343 control (winner's mesh) | 92.3 | 1 | -5.0 / +1.0 / +5.7 | 0.33 -> 0.99 |
| 0211 band 1 (z 7700-9200), 8 windings | 88.5-94.2 | 1 each | about -6 / 0 / +6 | -0.04-0.10 -> 0.90-0.97 |
| 0211 band 2 (z 4700-6200), 4 windings | 93.1-94.3 | 1 each | about -6 / 0 / +6 | -0.10-0.11 -> 0.85-0.93 |
| 0211 band 3 (z 9500-11000), w028 | 91.8 | 1 | -6.1 / -0.4 / +5.5 | 0.05 -> 0.92 |

The "after" excess is high **by construction**: the snap moves the surface to the density peak that this metric measures. The real check is
whether snapping keeps a known signal. On the PHerc 343 control (4-model ensemble, reading side = reverse, AUC letters vs rest), the snapped
render gives 0.623 (d=0) and 0.634 (d=-2), against 0.640/0.647 on the winner's own mesh (d=0/+1). The shifted-box nulls are 0.58-0.61 on both. The label-free
asymmetry also survives (coherence forward/reverse 0.296/0.330, model agreement +0.22/+0.58, p99(z) 2.31/2.97). So the snap does not destroy
real ink. It also does not sharpen the control, which is weak either way. Figures: `docs/figures/pherc0211_snap/`
(`w040_mesh_vs_snapped_layer33.png`: dark inter-sheet gaps under the fitted mesh, top; continuous fibre grid after snapping, bottom).

**seed42 screen on the snapped stacks, d=0** (`data/spiral/snap_ens/screen_seed42.csv`): control reverse/forward coarse_std 1.27. 0211:
w020 0.96, w030 1.17, w035 1.05, w040 0.996, w045 0.99, w050 1.13, w055 0.89, w065 0.96 (band 1); w025 0.99, w040 0.88, w055 0.96,
w070 0.96 (band 2); w028 1.01 (band 3). None reaches the control.

**4-model ensemble on snapped windings, d=0** (coherence fw/rv, model agreement fw/rv, p99(z) fw/rv; control 0.296/0.330, +0.22/+0.58, 2.31/2.97):

| winding | coherence | agreement | p99(z) |
|---|---|---|---|
| w030 | 0.290 / 0.310 | +0.26 / +0.36 | 2.60 / 2.45 |
| w045 | 0.315 / 0.302 | +0.46 / +0.40 | 2.43 / 2.44 |
| **w050** | 0.271 / 0.290 | **+0.11 / +0.52** | **1.96 / 3.39** |
| w050 at d=-8 (gap) | 0.270 / 0.280 | +0.25 / +0.28 | 2.89 / 2.55 |
| w050 at d=+8 (gap) | 0.267 / 0.276 | +0.18 / +0.19 | 2.05 / 2.17 |
| w050 unsnapped mesh | 0.283 / 0.303 | +0.22 / +0.53 | 2.19 / 3.61 |
| w055 | 0.288 / 0.281 | **+0.51 / +0.17** | 2.81 / 2.27 |

**w050 is a new, local lead, not text.** Its whole-map reading-side statistics peak on the sheet and are back to baseline at +-8 layers (but see
the follow-up below: inside the hotspot box the response is still there at d=-8). It is also present on the unsnapped
mesh, but w050 was never in the earlier ensemble set (20/30/35/40/45/55/65). However:

1. It comes from one cloud of blobs about 9 x 2 mm (960 x 210 render px) at the top-left of the render (render px y 230-440, x 780-1740; scroll xyz about
   (3140-3900, 4220-4280, 7775-7953), next to the band's z=7700 edge). With that box masked, agreement is +0.13/+0.18 and p99 2.08/2.32: no asymmetry.
2. The maps are round blobs, not stroke rows (`ensemble_z_w050_senseCW_d0.png` vs `ensemble_z_control_343_d0.png`). The CT under it shows fibres,
   cracks and a dense inclusion (`w050_hotspot_ct_layer33.png`).
3. The same statistic swings just as far the *other* way on the neighbouring winding (w055: forward +0.51, reverse +0.17). One-sided agreement of
   this size happens without ink on this scroll, so the label-free test cannot call w050 on its own.
4. It is at least 81 voxels (~0.8 mm) from the Reader v2 `z7920_w040` lead, so the two are unrelated.

Next step, if pursued: fit a z-band centred on z~7850 (so the hotspot is not at a band edge), snap, and run the ensemble at d -4..+4 in both
senses. Letters would need rows of stroke-shaped responses across more than one winding. Done 2026-10-24; see the next section.

**What this changes.** The 0211 negative no longer rests on meshes that are off the sheet: on snapped stacks of 13 windings in three bands the
result is the same (no control-like signature, apart from one local blob cluster). It is still not a proof of absence. The snap is our own,
the meshes come from 600-3000-step fits (not 30k), only one depth was screened, and the label-free thresholds come from one weak control.

## w050 hotspot follow-up (2026-10-24): new band centred on the hotspot

**Verdict: blob cluster, not text.** The hotspot is real, in the sense that it reproduces on an independent fit. It is one-sided, but it stays a
compact, roughly isotropic cluster on one sheet. It does not form rows along the writing direction, and neither neighbouring sheet shows it or
any reverse-side signature. No letters are claimed.

**Setup.** New band z 7100-8600 (tag `b6k3`, half-density tracks as in bands 3-5, 3000-step fit, satisfied track points 50.6 %), so the
hotspot (z 7755-7980) is ~650 voxels inside the band instead of at its edge. `scripts/spiral0211/locate_xyz.py` puts the old hotspot on new
winding **w048**: within 0.6-2.1 voxels of all five corner/centre points. Here the fitted windings are only ~10 voxels apart, inside the
+-8-layer snap window, so w047/w049 could snap onto the same sheet. The neighbour controls are therefore **w046 and w050**, ~19-22 voxels away.
Their snapped sheets lie a median 20-22 voxels from w048's (p5 >= 9; sheet spacing ~16), so they are different sheets. The new w048 snapped sheet
is the same physical sheet as the old band-1 w050 one: meshes 1-3 voxels apart, normals parallel, snap depths within 3.4 layers. Each mesh was
cropped (`crop_tifxyz.py`) to a 2.5 cm strip along the winding around the hotspot x the full band height (3.2 cm^2). It was then rendered (66 layers;
logs end at tile-row 100 %, 294-303 MB), snapped (92-94 % assigned, one patch) and run through the 4 models at d -4,-2,0,+2,+4 (`b6_chain.sh`).
The snapped PHerc 343 control was run at the same depths.

**Mask problem in the earlier 4-model numbers.** Hecate outputs zero wherever the masked 0211 CT has small holes (54-67 % of each 0211 map; 23 %
on the control). The 4-model "all valid, eroded 30 px" mask used by `ensemble_agree.py` therefore keeps only **8-12 % of a 0211 map** (29 % of the
control), so the 4-model tables above describe small fragments. It can mislead: w046 here gives 4-model agreement +0.23/+0.41 (fw/rv) on its 8 %
mask, but +0.25/+0.27 with the three villa models. The numbers below therefore use **seed42 + seed43 + dense_native** (mask 86-88 %). Hecate maps
still exist for every run. `scripts/spiral0211/row_blob_score.py --models seed42,seed43,dnative`; all rows are in `results/pherc0211_hotspot/rowblob_3model.csv`.

**Statistics, range over d = -4..+4, forward / reverse (reverse = reading side for the control).** agree = mean pairwise model correlation;
top1 = share of hot area (smoothed z > 1.5) in the largest cluster; acx/acy = autocorrelation at a 1 mm shift along / across the writing
direction; box = mean z in the hotspot box minus map mean.

| strip | agree | p99(z) | top1 % (rv) | acx / acy (rv) | box (rv) |
|---|---|---|---|---|---|
| 343 control (letters) | 0.20-0.24 / **0.35-0.51** | 2.23-2.42 / **2.64-3.18** | 15-21 | 0.26-0.29 / 0.05-0.14 | - |
| 0211 w048 (hotspot sheet) | 0.23-0.26 / **0.33-0.44** | 2.40-2.46 / 2.68-2.75 | **37-46** | 0.38-0.43 / 0.32-0.40 | **+1.03-+1.17** |
| w048 with box masked | 0.24-0.26 / 0.25-0.39 | 2.39-2.44 / 2.39-2.75 | 11-23 | 0.29-0.41 / 0.16-0.30 | - |
| 0211 w046 (one sheet in) | 0.21-0.26 / 0.26-0.29 | 2.34-2.58 / 2.42-2.53 | 11-19 | 0.22-0.29 / 0.10-0.16 | +0.22-+0.53 |
| 0211 w050 (one sheet out) | 0.20-0.27 / 0.23-0.24 | 2.39-2.64 / 2.40-2.64 | 22-39 | 0.21-0.26 / 0.17-0.22 | +0.02-+0.14 |
| band-1 w050, d=0 (edge mesh) | 0.19 / 0.33 | 2.33 / 2.97 | 20 | 0.36 / 0.23 | +1.91 (d=-8: +1.53, d=+8: 0.00) |
| band-1 w055, d=0 (no lead) | 0.35 / 0.19 | 2.79 / 2.32 | 8 | 0.25 / 0.05 | - |

What separates the hotspot from the control:

1. **One clump, not rows.** On the control's reading side a row of stroke-shaped responses runs along the writing direction at every depth
   (`docs/figures/pherc0211_hotspot/ens3_depth_control_343.png`). Its largest cluster holds 15-21 % of the hot area, and the response is
   2-5x more correlated along the writing direction than across it. On w048 one zone around and above the box holds 37-46 % of the hot area.
   The autocorrelation is isotropic (acx ~ acy ~ 0.4), and the zone is made of round blobs with no baseline or row spacing
   (`ens3_depth_b6_w048_hot.png`, `w048_hotzone_ct_vs_ens.png`).
2. **No neighbour support.** Neither w046 nor w050 shows a reverse-side agreement excess (+0.26-0.29 vs +0.21-0.26, and +0.23-0.24 vs
   +0.20-0.27), or rows (`ens3_depth_b6_w046_hot.png`, `ens3_depth_b6_w050_hot.png`). w046 has one isolated bright blob near the box, but it is
   2.2 mm (238 voxels) from w048's peak, so it is not the same object.
3. **The asymmetry size is not special.** w048's reverse-forward agreement gap (+0.10 to +0.20) is matched by the opposite gap on band-1 w055
   (+0.16 the other way) where there is no lead. With the box masked, w048 drops to +0.01 to +0.15.
4. **Depth.** Inside the box the response is flat from d=-4 to +4 (+1.03-+1.17). On the band-1 mesh it was still +1.53 at d=-8 and gone at +8.
   So it extends at least 8 layers (75 um) to one side of the sheet. The control's response is also broad over -4..+2, so this alone does not
   rule out ink, but it does not point to a thin surface layer either.
5. **Not a CT-brightness effect.** The correlation between the ensemble z and the local CT mean is -0.05 on w048, and the box is slightly darker
   than the map (-1.9 grey levels, sd 5.0).

**What was established.** The band-edge explanation is ruled out: the cluster reappears on a mesh from a different fit, ~650 voxels from the
band edge. So something on this sheet at scroll xyz ~(3100-3950, 4210-4300, 7750-7980) drives one-sided model agreement. It has none of the
spatial structure that the same models produce on real letters, and the adjacent sheets do not share it. The one remaining 0211 ensemble lead
is closed as "blob cluster, not text".
Limits: one control (343, letters ~4 mm), half-density tracks, 3000-step fit, our own snap, 3.2 cm^2 strips rather than whole windings.
