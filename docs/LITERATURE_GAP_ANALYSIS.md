# Literature / community gap analysis (2026-10-29)

Web check of what others have done on the problem we are working on (ink in PHerc 0211 at 9.362 um, scan 20250821151803). Sources are the scrollprize.org pages and repos linked below; anything not confirmed on a page is marked unverified.

## Bottom line

- Nobody has published letters from PHerc 0211. The only other full run of the official First Letters workflow on it (bnleft/first-light-pherc0211, Bryant Le, Sept 2026 progress prize) reports the same negative: "didn't see any ink in the window", control (PHerc 0139) reproduced, 0.019% of pixels over threshold, no letters claimed. That is independent agreement with our band 1-5 screens, Reader v2 run and snapped re-screen.
- Our pipeline matches what the winners used (spiral fit, snap, flatten, VC3D render, ink_9um + Hecate + dense-native + reader-v2 ensemble, positive control on 343). We are not missing a published method that would be expected to flip 0211 to positive.
- Scrollprize's own "Open Problems" page lists "no ink" vs "ink the pipeline cannot use yet" as unsolved, and says working ink detection exists only for PHerc 1667 and Paris 4.
- Only one 0211 scan is public (9.362 um, 113 keV). No 2.4 um 0211 volume is listed on its data-browser page. Ink on 1667 and Paris 4 was made visible by the optimised ESRF scan protocol, not only by better models.

## Methods we have not used, ranked by expected value

| # | Method | Source | What it would change for us | Cost |
|---|--------|--------|-----------------------------|------|
| 1 | Re-check Paris 4 with the official `scrollprize/ink_3d_dino_guided` (3D nnU-Net-style, MIT, 256^3 patches) | HF model card; villa repo | Our Paris 4 evaluation gave chance AUCs (see paris4 notes). The official model is the known-working reference there; if it also scores chance on our data, our eval/data handling is wrong; if it works, our eval harness is the gap. Validates the pipeline, not 0211. | Low-medium (download ckpt, `vesuvius` pkg, 1 segment) |
| 2 | Pseudo-label loop (train on fragments, predict sealed scroll, keep conservative labels, fine-tune, repeat) | PHerc 1667 work, Open Problems page | This is how 1667 was read. Needs seed ink to start; our 0211 maps show none, so it would amplify noise unless ProbeGuide-style filtering is used. | High |
| 3 | ProbeGuide: filter pseudo-labels by DINO feature similarity | slchicoineau-creator/probeguide | Reduces the false-positive risk of #2. | Medium, only with #2 |
| 4 | ScrollFiesta mesher (snaps to papyrus, makes constraints for spiral fit) | Hob3rMallow/scrollfiesta_public | Surface placement is the stated main bottleneck; our fit + snap_render is homemade because snap_meshes.py is not public. Better surfaces could remove the "mesh may be off-sheet" caveat on our negatives. | Medium |
| 5 | Surface-topography ink cue (morphology of written regions) | arXiv 2603.27698 (Angelotti et al.) | Signal degrades with lateral resolution; paper is on optical profilometry, no code link found. 9 um CT probably too coarse. Unverified for CT. | Research only |
| 6 | Different target scroll | prizes page: 21 eligible First Letters volumes | If 0211 holds no recoverable ink at 9 um, effort is better spent on a volume with a better scan. We have not checked which eligible volumes have 2.4 um data. | Needs data-browser check; user declined a PHerc 1203 download earlier |

Already covered by us, no further gain expected: reader-v2 (Domenico Russo), Hecate, dense-native (Nieuwlaar), ensembling (Alex Clark), depth-window tuning (Camillo), checkpoint benchmarking.

## Cautions

- A third-party repo (axiosdevs/herculaneum-scroll-tools) was reported by search to claim 43.3% "phantom voxels" for 0211. Could not open the repo (404) and the method is unknown, so it is not used as evidence.
- ibarapascal/ink-disagree argues evaluation against the 2026-09-18 ink labels depends strongly on which surface is scored; relevant to how we read any AUC. Not independently checked.
- First Letters requires 10 legible letters in 4 cm^2 plus false-positive mitigation and held-out validation; a candidate has to survive our existing controls (shifted-surface, reversed direction, spatial null).

## Sources

- https://scrollprize.org/2026_open_problems
- https://scrollprize.org/winners
- https://scrollprize.org/prizes
- https://scrollprize.org/tutorial5
- https://scrollprize.org/data_browser/PHerc0211
- https://github.com/bnleft/first-light-pherc0211
- https://huggingface.co/scrollprize/ink_3d_dino_guided
- https://arxiv.org/abs/2603.27698
- https://arxiv.org/abs/2606.29085


## Result: official Paris 4 3D model on our w00 tile (2026-10-29)

Script `scripts/probes/ink3d_official.py` (infer + score), output `results/ink3d_official/w00.json`. Checkpoint `scrollprize/ink_3d_dino_guided` `ckpt_78k_fullsup.pth` (EMA weights, sha256 5a148c2c...b303 matches the published hash), run with 256^3 patches, 50% overlap, per-patch percentile normalisation on our 2.4 um crop (`data/real_scans/paris4/native_crop.h5`, 1536x1408x768), no surface needed. Scored at the 3.01 M supervised w00 label pixels placed in 3D by the segment's x/y/z grid, same label-shift null as `ink_spatial_null.py` (200 shifts >= 400 px).

| sampling | AUC |
|---|---|
| value at the label's own voxel, dz 0 | 0.641 (0.58-0.67 over dz -24..+24) |
| max in a 9^3 neighbourhood, dz 0 | **0.880** (0.85-0.88 over dz -24..+24) |

Null 5-95% for the best setting: 0.42-0.60 (our homemade model's null was 0.23-0.79); observed 0.88 is outside all 200 shifts (p < 0.005).

What it means:
- Our label/coordinate/tile handling and the null test work: a model that does see Paris 4 ink gets a clearly significant AUC through the same code. The earlier chance results come from our fragment U-Net on 2.4 um scroll data, not from the harness.
- Caveats: in-sample. Paris 4 labels were part of this checkpoint's training, so 0.88 shows the harness is sound, not that the model generalises. The 9^3 max-pool and the dz sweep were chosen after seeing scores (14 settings; flat 0.85-0.88, so little selection effect). Label-pixel AUC on blobby labels still says nothing about letter legibility.
- The model's response is a thick 3D shell around the sheet (single-voxel AUC 0.64, 9^3 max 0.88), which is why surface-offset (~8 voxels, see INK_EVALUATION.md) matters less for it than for a surface-conditioned 2D model.
- Not done: other four labelled segments (w01-w03, w05; crops exist under data/real_scans/paris4/*), edge-band test, and a held-out check (none of the released checkpoints excludes Paris 4).
- Does not transfer to 0211: this model is trained on 2.4 um Paris 4 and needs ink visible in 3D; 0211 is 9.362 um.

### All five labelled Paris 4 segments (w00-w05), same model, setting fixed up front (9^3 max, dz 0)

`scripts/probes/ink3d_official_all.sh`; per-segment JSON in `results/ink3d_official/`. Label-shift null: 200 shifts >= 400 px. w00 was scored before the edge-band code was added (no edge-band row).

| segment | label pts | single-voxel AUC | 9^3-max AUC | null 5-95% | p |
|---|---|---|---|---|---|
| w00 | 3.01 M | 0.641 | **0.880** | 0.43-0.60 | <0.005 |
| w01 | 2.31 M | 0.570 | **0.725** | 0.37-0.59 | 0.005 |
| w02 | 1.43 M | 0.615 | **0.711** | 0.36-0.68 | 0.03 |
| w03 | 1.81 M | 0.622 | **0.705** | 0.36-0.61 | <0.005 |
| w05 | 2.47 M | 0.761 | **0.840** | 0.37-0.59 | <0.005 |

Edge band (supervised points within B canvas px of a label edge; `band` raw, `hp` high-passed at sigma 48; p two-sided vs the shifted-label null):

| segment | B=16 band / hp | B=32 band / hp | B=64 band / hp |
|---|---|---|---|
| w01 | 0.52 (p .40) / 0.48 (.34) | 0.54 (.30) / 0.46 (.28) | 0.58 (.20) / 0.46 (.39) |
| w02 | 0.55 (.13) / 0.53 (.42) | 0.62 (.07) / 0.57 (.24) | 0.70 (.05) / 0.61 (.17) |
| w03 | 0.52 (.33) / 0.51 (.61) | 0.55 (.17) / 0.51 (.75) | 0.62 (.08) / 0.52 (.70) |
| w05 | 0.58 (.02) / 0.52 (.39) | 0.64 (.02) / 0.55 (.10) | 0.72 (.02) / 0.57 (.06) |

Reading:
- The official model is significantly above the shifted-label null on all five segments (pooled AUC 0.70-0.88), through our own label/geometry/null code. The earlier chance results on Paris 4 therefore come from our fragment U-Net, not from the harness. In-sample caveat as above; the lower AUCs on w01-w03 may also reflect label/geometry misalignment (labels were drawn on an older scan).
- The edge-band test is much weaker: even this in-sample reference model is at chance on w01, w03 and every high-passed band, and nominally significant only on w05 raw bands (p 0.02, not after high-pass). So the edge-band test cannot detect a working ink model on these labels, and our earlier edge-band "no signal" results (docs/INK_EVALUATION.md, 5 segments with our own model) are uninformative about whether ink was present. They should not be quoted as evidence of absence.
- Same logic limits the 0211 negatives only where they relied on label-based tests; the 0211 work uses unlabeled signatures and the 343 positive control, which are unaffected.
