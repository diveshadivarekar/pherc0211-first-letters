"""Label-free ink screen on snapped renders (snap_render.py output): ink_9um seed42, forward + reverse, per depth.

Same statistics as screen_L1.py (coarse_std, band_std, coherence of the prediction map over eroded valid pixels), at level 0 on
the snapped stack. On the PHerc 343 control the reading side (reverse) has clearly higher coarse_std than the blind side
(forward); a winding is flagged when reverse/forward >= --ratio (default 1.3, below the control's 1.5-2.3).
Run with D:/divesh-github/inkenv/Scripts/python.exe.
Usage: snap_screen.py <out_dir> <csv> <depths e.g. 0,-2> <label=render.zarr> [<label=render.zarr> ...]
"""
import csv, os, subprocess, sys
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter, binary_erosion

out, out_csv, depths = sys.argv[1], sys.argv[2], sys.argv[3]
items = [x.split("=", 1) for x in sys.argv[4:]]
PY = "D:/divesh-github/inkenv/Scripts/python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out, exist_ok=True)


def stats(a):
    valid = a > 0
    v = binary_erosion(valid, iterations=30)
    if v.sum() < 5000:
        return None
    af = a.astype(np.float32)
    band = gaussian_filter(af, 1.5) - gaussian_filter(af, 12)
    gy, gx = np.gradient(gaussian_filter(af, 2.0))
    jxx, jyy, jxy = (gaussian_filter(q, 25) for q in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt(np.maximum((jxx - jyy) ** 2 + 4 * jxy ** 2, 0)) / (jxx + jyy + 1e-6)
    return dict(valid_px=int(v.sum()), coarse_std=float(af[v].std()), band_std=float(band[v].std()), coherence=float(coh[v].mean()))


done = set()
if os.path.exists(out_csv):
    done = {(r["label"], r["d"]) for r in csv.DictReader(open(out_csv))}
new = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    wr = csv.writer(f)
    if new:
        wr.writerow(["label", "d", "fw_coarse_std", "rv_coarse_std", "rv_fw_ratio", "fw_band_std", "rv_band_std", "fw_coh", "rv_coh", "valid_px"])
    for label, render in items:
        todo = [d for d in depths.split(",") if (label, str(int(d))) not in done]
        if not todo:
            continue
        native = "8.64" if "control" in label else "9.362"
        subprocess.run([PY, "-u", f"{HERE}/ensemble_run.py", render, out, label, native, ",".join(todo), "seed42"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for d in todo:
            base = f"{out}/{label}__seed42__d{int(d)}"
            if not (os.path.exists(base + ".tif") and os.path.exists(base + "_reverse.tif")):
                wr.writerow([label, int(d)] + [""] * 8); f.flush(); continue
            s_f, s_r = stats(tifffile.imread(base + ".tif")), stats(tifffile.imread(base + "_reverse.tif"))
            if not (s_f and s_r):
                wr.writerow([label, int(d)] + [""] * 8); f.flush(); continue
            wr.writerow([label, int(d), round(s_f["coarse_std"], 2), round(s_r["coarse_std"], 2), round(s_r["coarse_std"] / s_f["coarse_std"], 3),
                         round(s_f["band_std"], 2), round(s_r["band_std"], 2), round(s_f["coherence"], 3), round(s_r["coherence"], 3), s_r["valid_px"]])
            f.flush()
            print("done", label, d, flush=True)
