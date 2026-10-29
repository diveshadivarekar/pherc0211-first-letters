# Reader v3 pipeline (draft, 2026-10-28)

"Reader v3" here is **our own pipeline revision**, not a new model. The public Reader v2 repo has no v3; the
checkpoint is still `reader-v2-step040000.pth`. The ink SDK is the villa `vesuvius.ink_detection` package already
used for inference (`infer`, `koine_machines` optional, not installed).

Entry point: `scripts/reader_v3/run_reader_v3.py` (supersedes `scripts/reader_v2/chain*.sh`).

| v2 | v3 |
|---|---|
| per-chain shell scripts, `CENTER` set by hand | one script; mesh layer = depth // 2 |
| fixed offsets only; sheet tracking was a separate manual step | `--sheet mesh/A/B/Bedge` tracks the sheet per pixel |
| scoring and shape stats in separate scripts | score + vertical/horizontal component share in the same run |
| control checked by hand | `--control` re-runs the 343 control first and aborts if it does not flag |
| no provenance | `manifest.json` (checkpoint sha256, args, villa HEAD) per surface; one `score_all.csv` |
| concurrent jobs crashed Windows | strictly sequential, resumable (finished maps are skipped) |

Example:

```bash
D:/divesh-github/inkenv/Scripts/python.exe scripts/reader_v3/run_reader_v3.py \
  data/spiral/render/w045_cw600.zarr own_w045 --offsets 0 7 -7 \
  --control data/control_343/render/concat_crop.zarr --control-name ctrl343 --control-offset 1
```

Status: smoke test on the control only; no new 0211 results. The 0.008 flag threshold and the two-orientation
caveat from `PHERC0211_READER_V2.md` still apply.
