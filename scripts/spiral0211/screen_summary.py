"""Summarise the level-1 screen. Reference (PHerc 343 control at level 1, ink present):
reverse coarse_std ~30, band_std ~6.0, coherence ~0.39 | forward 13-19, 2.9-3.9, 0.26-0.30."""
import csv, sys
import numpy as np
rows = {}
for r in csv.DictReader(open(sys.argv[1])):
    if r["coarse_std"] == "":
        continue
    rows.setdefault(int(r["winding"]), {})[r["dir"]] = (int(r["valid_px"]), float(r["coarse_std"]), float(r["band_std"]), float(r["coherence"]))
print("wind  valid_px   fw: cstd bstd coh   rv: cstd bstd coh   rv/fw cstd  flag(ink-like: rv cstd>=26 & coh>=0.33 & ratio>=1.3)")
n_flag = 0
for w in sorted(rows):
    f, v = rows[w].get("forward"), rows[w].get("reverse")
    if not f or not v:
        continue
    ratio = v[1] / f[1]
    flag = v[1] >= 26 and v[3] >= 0.33 and ratio >= 1.3
    n_flag += flag
    print(f"{w:4d} {f[0]:9d}   {f[1]:5.1f} {f[2]:4.1f} {f[3]:.3f}      {v[1]:5.1f} {v[2]:4.1f} {v[3]:.3f}    {ratio:4.2f}   {'<<<' if flag else ''}")
allf = np.array([[rows[w]['forward'][i] for i in (1, 2, 3)] for w in rows if 'forward' in rows[w] and 'reverse' in rows[w]])
allr = np.array([[rows[w]['reverse'][i] for i in (1, 2, 3)] for w in rows if 'forward' in rows[w] and 'reverse' in rows[w]])
print(f"\n{len(allf)} windings; flagged {n_flag}. median fw {np.median(allf,0).round(2)} rv {np.median(allr,0).round(2)}; max rv {allr.max(0).round(2)}")
