# PHerc. 0846A window W1 (survey candidate 0, 2026-10-12) - feasibility only, no ink demonstrated

Source: PHerc. 0846A, 2.403 um / 77 keV volume `20260319102732`, native crop z 10240-11264, y 13568-14080, x 20224-21760
(`data/real_scans/pherc0846a/native_crop_W1.h5`, not committed) with the `m7-L2` surface prediction (9.6 um voxels).
Chosen by the level-3 survey (`scripts/survey_regions.py`, body fraction >= 0.727, bright fraction <= 0.006, highest planarity + regularity).
The crop is 82 % papyrus-band voxels, 15 % void, 0.6 % saturated debris (W0 was mostly void).

**Not submission evidence.** Faces are 5-9 mm^2, plane-fit height fields (not isometric flattenings), no tifxyz mesh; the eligibility of the
2.4 um volume for First Letters is unconfirmed.

## Faces (`scripts/probes/face_from_prediction.py`)
| face | area | flatness | raw-surface registration | note |
|---|---|---|---|---|
| 0 | 9.1 mm^2 | 0.29 | peak at -10 (window edge) | multi-sheet blob, texture changes between offsets; unusable |
| 1 | 7.5 mm^2 | 0.15 | peak at +10 (window edge) | poor registration; not examined |
| 2 | 6.4 mm^2 | 0.12 | peak at -5 | best behaved: stable across offsets, fibre grid, voids; `face_2.png` |
| 3 | 5.2 mm^2 | 0.24 | peak at -5 | `face_3.png` |
At native scale the layers here are thin and heavily crumpled, so a plane-fit height field is a poor model; the planarity that selected the block
is measured at 19 um and does not transfer to the faces.

## Ink-model check (`scripts/probes/ink_on_faces.py`, same fragment U-Net and +-30 voxel shifted controls as W0; `ink_stats.json`)
Share of valid pixels with P(ink) > 0.5 (about 1 M valid pixels per surface, so the controls are well sized here):

| surface | normal | flipped |
|---|---|---|
| face 2 | 0.2 % | 9.0 % |
| face 2 +30 / -30 | 4.3 % / 7.7 % | 1.4 % / 0.0 % |
| face 3 | 0.0 % | 0.0 % |
| face 3 +30 / -30 | 4.2 % / 3.6 % | 6.5 % / 2.9 % |

* In the normal orientation the model hardly fires on the real surfaces and fires more on the shifted copies: the reverse of W0. The response is not
  tied to the surface and changes sign between windows.
* The one face-specific response (face 2 flipped, 9.0 % vs 1.4 % / 0.0 %) is a few smooth round blobs of 0.29 / 0.12 / 0.09 mm^2 near the top rim
  of the face (`ink_check_face2_flip_UNVALIDATED.png`: page | P(ink) | P > 0.5), not strokes.

Conclusion: still no evidence of ink. The model has no ground truth to be checked against on this scroll, and nothing here supports any letter.
