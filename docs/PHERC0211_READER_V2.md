# PHerc 0211 - Reader v2 ink-detection test (2026-10-15)

Scan 20250821151803 (9.362 um, 113 keV). Model: [domenicor046/reader-v2](https://github.com/DomRusso2/reader-v2),
checkpoint `reader-v2-step040000.pth` (138,382,373 bytes, sha256
`654ec5acec2b6c4788d9cb326e2f0b8c2c730584949a934ef775db7f6ab5d3a6`, MIT).

## Verdict

**No legible letters were found. The requirement of >= 10 characters inside 4 cm^2 is not met, and no characters,
sentence, OCR or translation are claimed.**

* Positive control passes: on the PHerc 343 winner mesh, Reader v2 gives a clean row of letter blobs on the reading
  side (text-screen 0.021, inside the model author's own "known text" band 0.010-0.032).
* 17 PHerc 0211 surfaces (about 100 cm^2 of rendered surface, 13 from our own spiral-fit meshes, 4 community meshes),
  evaluated at the mesh: **0 flagged**. Highest 0.0064 (our crude w045); every value is below the 0.008 flag
  threshold and most are inside the author's blank/false band (0.0003-0.0057).
* One unexplained lead: mesh `z7920_w040` shows a one-sided, control-like response **+10..+13 voxels (about 94-122 um)
  off the mesh** (screen 0.028; 0.034 after tracking the neighbouring sheet). Its spatial pattern is rectangular blocks
  beside a damaged, delaminated zone, not rows of letter-sized strokes. Re-segmenting that sheet did not change this.

## What was run

* Same villa `vesuvius.ink_detection.inference.infer` entry point and `inkenv` already used in this repo
  (`--overlap 0.5 --blend-mode hann --direction both`, 17-layer window). Reader v2 states it is a drop-in for
  `ink_9um`; the checkpoint loaded with `missing_keys=0 unexpected_keys=0`. Not run: the `koine_machines` package
  and the `infer_best_quality` / Hecate modes (not installed; they are optional variants).
* Screen = reader-v2's calibrated `textscore2.score2`: fraction of valid pixels that are confident ink in one
  direction (>= 180), not ink in the other (<= 110), not on a CT void, in stroke-sized components. Both
  orientations are computed; the winner's reading side was the reverse map.
* Surfaces:
  * own: 13 windings from the spiral-fit bands (`data/spiral/render`, 66-layer renders, mesh at layer 33);
  * community: 4 of the 5 requested surfaces of the `pscamillo/vesuvius-eligible-meshes` commit `0c966f8f`, as
    rendered in `rodriguescarson/eligible-scroll-atlas-renders` (CC BY-NC 4.0, 31 layers, mesh at layer 15,
    `--flip-normals`). `z9712_w080` has a zero-filled root `.zattrs` and a chunk that fails zstd decoding in the
    downloaded tar; it was excluded;
  * control: PHerc 343 winner mesh crop (`data/control_343`).

## Results at the mesh (d = 0)

Full table: `docs/figures/pherc0211_reader_v2/score_table.md`, raw numbers `score_all.csv`.

| surface | valid mm^2 | screen fwd | screen rev | flag |
|---|---|---|---|---|
| 343 winner mesh (control, d=+1) | 317 | 0.00000 | **0.02093** | yes |
| own w045 (600-step fit) | 582 | 0.00638 | 0.00431 | no |
| atlas z7920_w040 | 380 | 0.00502 | 0.00115 | no |
| atlas z6720_w080 | 838 | 0.00461 | 0.00125 | no |
| own w040 (senseCW) | 519 | 0.00457 | 0.00041 | no |
| atlas z9120_w080 | 857 | 0.00393 | 0.00022 | no |
| atlas z7312_w080 | 836 | 0.00390 | 0.00275 | no |
| 11 further own windings | 260-903 | <= 0.0032 | <= 0.0029 | no |

The four atlas values match the reader-v2 author's published sweep (`results/textscore2_040000.csv`: 0.00502,
0.00462, 0.00392, 0.00392) to within 0.00002, so this setup reproduces their numbers.

## The z7920_w040 depth profile

Re-rendered the mesh with 66 layers (matches the atlas render, r = 0.99) and ran windows from -16 to +24 voxels
(`z7920_depth_profile.png`):

| offset (vox) | -16 | -10 | -7 | -4 | 0 | +4 | +7 | +10 | +13 | +16 | +19 | +22 | +24 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| reverse screen | .0046 | .0078 | .0035 | .0032 | .0012 | .0150 | .0220 | **.0283** | .0277 | .0196 | .0110 | .0042 | .0019 |
| forward screen | .0001 | 0 | 0 | .0018 | .0050 | .0022 | .0012 | 0 | .0001 | .0004 | .0002 | .0001 | .0002 |

* The reverse map's coarse std peaks at 35.6 at +10 (control reading side: 31.8, control other side: 13.5), with
  the forward side at 11-13. That is the control's statistical signature, and it is one-sided.
* Other surfaces at +-7 voxels (z6720, z7312, z9120) did not flag (max 0.0037), so this is specific to this mesh.

Why it is **not** called text:

1. Geometry. The response is rectangular blocks (x 0.5-7 mm and 10.5-17 mm, full 7 mm height) with straight
   vertical edges, not rows of 0.5-2 mm strokes. Of confident component area at +7..+13 voxels, 38-57% is
   vertically elongated and only 8-24% horizontally elongated; the control is 44% horizontal / 26% vertical.
2. Context. The left block sits on a delaminated, void-riddled stretch of the CT (layer 44, `z7920_left_zoom.png`)
   bounded by a vertical seam at x = 7.5 mm. The reader-v2 author reports that the flags on their own sweep were
   surface crossings and crushed zones.
3. At native resolution (CLAHE) the CT at +11 voxels shows fibre grid and voids, no letterforms.
4. No validation labels exist for 0211, and the control only shows the model works on a surface where text is known.

## Follow-up: sheet-tracked re-run (done, same day)

To rule out that a fixed offset straddled the wrong layer, the mesh was re-rendered with 100 layers
(mesh at layer 50, correlation 1.0 with the 66-layer render) and the neighbouring sheet was tracked per pixel
from the smoothed CT depth profile (`scripts/reader_v2/track_sheet.py`). Across the mesh footprint the mesh sits
about 2 voxels past sheet A's density peak, sheet B's near face is at +11 (median, p5-p95 +6..+18) and B's peak
at +17 (+11..+28). 33-layer stacks centred on each tracked surface were then read by Reader v2:

| stack | offset vs tracked surface | screen fwd | screen rev | coarse fw/rv | vertical / horizontal share of area |
|---|---|---|---|---|---|
| B near face | 0 | 0.00023 | **0.03398** | 12.6 / 38.0 | 54% / 10% |
| B near face | +4 | 0.00049 | 0.02467 | 11.8 / 36.4 | - |
| B peak | -8 | 0.00000 | 0.03363 | 14.7 / 37.1 | - |
| B peak | -4 | 0.00007 | 0.02997 | 11.3 / 35.8 | 44% / 10% |
| B peak | 0 | 0.00000 | 0.01935 | 11.5 / 29.6 | 32% / 32% |
| A peak (own sheet, control) | 0 | 0.00727 | 0.00316 | 31.2 / 19.9 | 0% / 52% (fwd map) |

Tracking made the one-sided response a little stronger (0.034, just above the author's top "known text" value of
0.032) but did not change its shape: still rectangular blocks (x 0.5-7 mm and 10.5-17 mm) with a vertical-dominant
component mix, and the hits cluster along long dark voids and the x = 7.5 mm seam that the tracked surface now
cuts through (`trackB_edge_d0.png`). No rows of letter-sized strokes, so still no characters.

Interpretation, stated with its uncertainty: the response is real and one-sided, and it cannot be rejected by the
reader-v2 screen alone, but its geometry and its location next to damaged material fit a structural
cause (delamination edges, inter-sheet gap) at least as well as ink. Telling those apart would need either a
labelled 0211 region (none exists) or a clean, undamaged page where the same pattern shows letter rows.

## Provenance and licence notes

* Community meshes and atlas renders are third party and CC BY-NC 4.0; check this against the prize's licensing
  rules before putting them in a submission. Our own spiral-fit meshes carry no such restriction.
* `z9712_w080` tar from the atlas is damaged at source (root `.zattrs` all zero bytes; zstd chunk error); the
  zero-filled file was backed up and replaced by `{}` to allow opening, but inference still failed.
* Running several inference jobs at once exhausted Windows commit memory and crashed some runs
  (`ArrayMemoryError`, DataLoader worker exit); all numbers above come from completed runs, re-run one at a time.

## Reproduce

```bash
bash scripts/reader_v2/chain1.sh   # control, own w045, five atlas surfaces at d=0 (needs data/atlas0211 untarred)
bash scripts/reader_v2/chain2.sh   # remaining own windings
bash scripts/reader_v2/chain5.sh   # idempotent sequential clean-up
bash scripts/reader_v2/analyze_all.sh   # scores + data/reader_v2/score_table.md
python scripts/reader_v2/make_figure.py <render.zarr> <prefix> <depth> <out.png>
python scripts/reader_v2/component_shape.py <render.zarr> <prefix> <depth>
```

Figures and score tables: `docs/figures/pherc0211_reader_v2/`.
