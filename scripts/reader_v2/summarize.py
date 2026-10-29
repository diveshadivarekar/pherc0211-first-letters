"""Merge the Reader v2 score CSVs into one markdown table (sorted by the larger of the two orientation screens).

Reader v2's own calibration (results/textscore2_040000.csv in reader-v2): known text s180 0.010-0.032,
blank / false 0.0003-0.0057, flag >= 0.008.
Usage: summarize.py <out.md> <score.csv> [<score.csv> ...]
"""
import csv, sys

out, srcs = sys.argv[1], sys.argv[2:]
rows = []
for s in srcs:
    rows += list(csv.DictReader(open(s)))
for r in rows:
    r["best"] = max(float(r["screen_fwd_180"]), float(r["screen_rev_180"]))
    r["side"] = "fwd" if float(r["screen_fwd_180"]) >= float(r["screen_rev_180"]) else "rev"
rows.sort(key=lambda r: -r["best"])
with open(out, "w") as f:
    f.write("| map | depth | valid mm2 | screen fwd | screen rev | max | flag (>=0.008) | coarse fw/rv | coherence fw/rv |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        mm2 = int(r["valid_px"]) * (9.362e-3) ** 2
        f.write(f"| {r['map']} | {int(r['depth']):+d} | {mm2:.0f} | {float(r['screen_fwd_180']):.5f} | {float(r['screen_rev_180']):.5f} | "
                f"{r['best']:.5f} ({r['side']}) | {'**YES**' if r['best'] >= 0.008 else 'no'} | {r['coarse_fw']}/{r['coarse_rv']} | "
                f"{r['coh_fw']}/{r['coh_rv']} |\n")
print(f"{len(rows)} rows -> {out}")
