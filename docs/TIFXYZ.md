# tifxyz export and volume-aware surface QA

`src/tifxyz/` writes this pipeline's flattened pages in the Vesuvius Challenge **tifxyz** format (so they open
in VC3D and work with the `vesuvius` tools) and audits any tifxyz surface *against the scan volume*.
It needs only numpy, scipy, tifffile and (for the volume checks) h5py - no OpenCV, no `vesuvius` package,
no compiled extension - and runs on Windows with a CPU.

## Why a volume-aware audit

Existing tifxyz auditors ([tifxyz-doctor](https://github.com/aviad12g/tifxyz-doctor), [windcheck](https://github.com/joe-carr-data/windcheck),
[flatcheck](https://github.com/abundantjoe/flatcheck)) look at the mesh alone. A sheet that has drifted off the papyrus, or a page that
is really a plane through several wraps, is a perfectly well-formed mesh, so geometry cannot see it (tifxyz-doctor
states that natural sheet switches are not covered). This module adds checks that use the volume and the scroll axis:

| Check | Inputs | Question it answers |
|---|---|---|
| `check_contract` | tifxyz dir | Files, shapes, `scale` sane *and consistent with the grid's real vertex spacing* (catches the step-written-as-scale mistake) |
| `geometry_report` | tifxyz dir | Validity, components, enclosed holes, edge length vs declared scale, shear, fold-overs (median-reference, robust to small flipped patches) |
| `radiality` | + umbilicus | Is the normal radial? Cells whose normal is far from the radius from the scroll axis cut *across* wraps (folds, ramps onto the next wrap, hole-filled bridges) |
| `radial_extent` / `estimate_wrap_spacing` | + umbilicus, surface prediction | How many wraps does the page span *within a z-slice*? (median per-slab radial spread / wrap spacing measured by casting rays through the surface prediction). A sheet that leans in radius along z counts as one wrap |
| `surface_support` | + surface prediction | Share of vertices on predicted surface; signed distance to the nearest one along the normal |
| `snap_to_prediction` | + surface prediction | Moves vertices onto the centre of the nearest predicted sheet (smoothed shift field) - label snapping |
| `ct_centering` | + raw CT | Prediction-free density-ridge contrast at the vertices. **Weak signal, see caveats** |

## Use

```bash
# 1. export M3 pages (coordinates are tile-local; --offset is the tile's origin in the whole scroll, z y x)
python scripts/export_tifxyz.py data/processed/native_surface/pherc0332_native_tile_flattened.h5 \
    --out data/processed/tifxyz/pherc0332 --offset 16384 7680 7680 --voxel-um 2.399 --stride 4

# 2. audit any tifxyz directory or folder of them (each extra input enables more checks)
python scripts/tifxyz_qa.py data/processed/tifxyz/pherc0332 \
    --umbilicus data/real_scans/pherc0332_umbilicus.json \
    --prediction data/real_scans/pherc0332_surface_pred_crop.h5 \
    --ct data/real_scans/pherc0332_native_crop.h5 \
    --snap-out data/processed/tifxyz/pherc0332_snapped \
    --png-dir data/processed/tifxyz/qa_png --report data/processed/tifxyz/qa_report.json
```

From Python: `from src import tifxyz as tx` - `tx.from_coord_map`, `tx.write_tifxyz`, `tx.read_tifxyz`, and the checks above.
The `--offset` matters: tifxyz loaders silently drop vertices with `z <= 0`, and tile-local coordinates reach below zero, so
`from_coord_map` refuses them rather than writing a surface that loads half empty.

## Format compatibility - what was verified

* The 50 exported PHerc. 0332 surfaces load with the **unmodified official `vesuvius.tifxyz.read_tifxyz`** (villa, run in an isolated
  environment); the official reader and this module's reader agree on validity and coordinates for 50/50 surfaces.
* The reader also handles the two community tifxyz segments published for PHerc. 0332 (float32-rounded `scale`, minimal `meta.json`):
  contract check passes with no findings.
* Not verified: opening the files in the VC3D GUI itself (no VC3D build here).

## Results on PHerc. 0332 (2.399 um, centre of the scroll, 50 pages from this repo's M2/M3)

Full numbers: [`tifxyz/pherc0332_qa_summary.json`](tifxyz/pherc0332_qa_summary.json), per page: [`tifxyz/pherc0332_qa_per_page.csv`](tifxyz/pherc0332_qa_per_page.csv).

1. **Some of the pipeline's largest pages are not single sheets.** Measured *within horizontal slices* (a sheet that merely leans in radius
   along z is still one sheet), 3 of 50 pages span more than two wraps (up to 6.1 for `layer_1`) and carry **16.8 % of all mesh vertices**; they are
   on predicted surface only 48 % of the time versus 81 % for the rest. The patch unwrap fits one plane through a wedge piece, so it cuts across
   wraps. A second, different problem: 5 pages that span about one wrap or less (21.5 % of the area) sit under 70 % on predicted surface - off the
   sheet rather than across wraps.
   Figure: [`figures/tifxyz_qa_layer_1.png`](figures/tifxyz_qa_layer_1.png) - panels are prediction support, signed drift, cross-wrap cells, fold cells;
   the banding in the first two is the mesh crossing sheets. Compare the well-behaved [`layer_10`](figures/tifxyz_qa_layer_10.png).
   The upstream fix is in [WRAP_SPLITTING.md](WRAP_SPLITTING.md).
2. **Control for the tile offset.** With the true offset 52 % of `layer_1`'s vertices sit on predicted surface (median drift 2.1 voxels);
   shifting the whole mesh by 40 voxels drops that to 14-24 % (drift 7-8 voxels). The prediction's mean profile along the normals has a sharp peak at the mesh.
3. **Snapping is a marginal gain here, not a fix.** Per-page means, before -> after: on-surface 88.3 % -> 90.5 % (multi-wrap pages 69 % -> 75 %), median shift
   0.7 voxel. These pages were cut *from* the prediction, so they start close to it, and snapping a plane through seven wraps just pulls its
   vertices onto whichever wrap is nearest - the right fix there is upstream (cut pages per wrap). The tool is meant for meshes from other sources
   (hand-traced, neural-traced) that may drift.
4. **The raw-CT check is inconclusive at this resolution.** A 2.4 um sheet is tens of voxels thick, the CT profile around the mesh is a broad
   hump, and `centred_fraction` sat at chance (0.21 vs 0.20). `ct_contrast` rose after snapping on only 28 of 50 pages. Do not read it as an accuracy.

## Limits

* Radiality assumes a roughly vertical scroll axis (horizontal radius from the umbilicus). Near the scroll centre wraps are crumpled: on the 16
  well-behaved pages here (at most 2 wraps, over 95 % on predicted surface) the median normal is still 28 degrees off radial (page-median of
  medians; 39 degrees on the multi-wrap pages), so the default 60-degree threshold separates pages only coarsely. Use `cross_wrap_fraction`
  comparatively (page vs page) rather than as an absolute error rate.
* **`wraps_crossed` is only meaningful for pages spanning a small angle (up to ~90 degrees).** The scroll is oval, so one wrap's radius changes along its length: the measure correlates 0.78 with
  angular span, and 100-150-degree pages read as multi-wrap while lying 98 % on predicted surface. Results carry `angular_span_deg` / `wraps_crossed_reliable`; the summary counts only reliable pages
  (the 40-degree wedge pages in the results above all qualify). For longer pages use on-surface and cross-wrap cells.
* `wraps_crossed` needs the surface prediction to estimate wrap spacing (one spacing per surface, measured at its median height). It is a
  within-slice radial spread, so a strip that winds a full turn also reads as about one wrap. A first version measured the spread over the whole
  mesh, which counted sheets that lean along z as multi-wrap and overstated the problem (16 pages / 54.5 % of the area instead of 3 / 16.8 %);
  `radial_spread_total` / `wraps_crossed_total` still report it.
* All checks are statistics of a mesh, not proof it is on the right sheet; there is no labelled set of natural sheet switches to measure
  precision/recall against.
* Surfaces from the older 2024 PHerc. 0332 tracings are in a different volume frame from the 2.4 um data, so volume checks do not apply to them.
