# Which First Letters scroll to attack (survey of the 22 eligible scrolls, 2026-10-10)

Scope: the 22 scrolls on https://scrollprize.org/prizes (First Letters Prizes, deadline 25 June 2027). Nothing here is an ink result.
`scripts/survey_scrolls.py` lists each scroll's volumes and published representations from the open-data bucket
(`data/processed/scroll_survey/s3_survey.json`, not committed).

## Measured (S3 listing, 2026-10-10)

| Scrolls | Volumes in the bucket |
|---|---|
| 0125, 0191, 0211, 0257, 0358, 0813, 0826, 0846B, 1545 | one: 9.362 um / 113 keV |
| 0175A, 0175B, 0268, 0306B, 0343, 0483A, 0483B, 0490A, 0490B, 0800, 1218 | one: 8.64 um / 116 keV |
| **0846A, 1203** | 9.362 um / 113 keV **and** 2.403 um / 77 keV |

Only 0846A and 1203 have a 2.4 um / 77 keV volume (the scan type this repo was validated against on Scroll 1 / PHerc 0332).
Their level-0 shapes are 15137 x 29587 x 29587 (0846A) and 15137 x 26493 x 26493 (1203); level 3 (19 um voxels) is 9.2 GB (0846A) and 6.3 GB (1203) compressed; level 4 is 1.45 GB and 0.97 GB.
Neither has any segments or ink predictions on the data browser. The surface prediction on the 2.4 um volumes is the `m7-L2` product,
i.e. computed at about 9.6 um voxels, so it is not finer than the 9 um prediction.

## Open question that decides the plan (not answerable from the public pages)

The prize page lists the **9.362 um** volume IDs as the eligible scan for 0846A (`20250728152254`) and 1203 (`20250820131727`) and is silent on
whether a submission may be built on the 2.4 um volume (`20260319102732`, `20260319130212`). The Grand Prize has an explicit "no data derived from
higher-resolution scans" rule; First Letters has none, and only the Paris 4 title prize says "any scan". **Ask the organizers (Discord) before
building a submission on the 2.4 um volumes.**

## Prior work (read from the public pages, not reproduced here)

* PHerc 0826 (113 keV), Miller and Muller, Aug 2026 progress prize: full First Letters workflow with the published `ink_9um` checkpoint; they report
  seeing no ink and make no letter claims. Repo: https://github.com/millerandmuller/first-light-pherc0826 (MIT).
* PHerc 0139 ink work (Nieuwlaar, KLAVIS and others): public 9 um ink checkpoints, labels transferred from 78 keV scans of the same scroll; reported
  held-out AUC ~0.84-0.95 depending on set; the author states the maps show "letter-shaped strokes, not readable lines".
  Repo: https://github.com/Nieuwlaar/ink9um-dense-native.

## Consequences for this repo

1. The fragment-trained U-Net in this repo was trained on 54 keV / 3.24 um data and has no demonstrated signal above a spatial null on Scroll 1
   (docs/INK_EVALUATION.md). It should not be used to claim ink on 113 keV scans.
2. For 113 keV scrolls the community checkpoints above are the baseline to compare against, not this repo's model.
3. PHerc 0211 specifically: 12 largest flattened pages from four native crops show papyrus fibres and no strokes (see the PHerc 0211 section of
   `data/real_scans/README.md`); that is a visual check only.
