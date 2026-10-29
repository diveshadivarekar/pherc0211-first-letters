"""Cheap whole-band ink-signature screen: render each fitted winding at level 1 (18.7 um), run ink_9um at d=0
(forward + reverse), and write per-winding texture statistics. Validated on the PHerc 343 control
(reverse/reading side >> forward side, peak near d=0). Run with D:/divesh-github/inkenv/Scripts/python.exe.

Usage: screen_L1.py <run_dir> <tag> <w_first> <w_last> [d]   (w_first may be a comma list; w_last then ignored)
"""
import csv, os, shutil, subprocess, sys
import numpy as np, tifffile
from scipy.ndimage import gaussian_filter, binary_erosion

run, tag = sys.argv[1], sys.argv[2]
WINDINGS = ([int(x) for x in sys.argv[3].split(',')] if ',' in sys.argv[3]
            else list(range(int(sys.argv[3]), int(sys.argv[4]) + 1)))
d = int(sys.argv[5]) if len(sys.argv) > 5 else 0
R = "D:/divesh-github/virtual-unwrap/data/spiral"
CK = "D:/divesh-github/ink9um/hybrid_3d2d-seed42/step-075000.pth"
VC = "C:/Program Files/VC3D/bin/vc_render_tifxyz.exe"
URL = "s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr"
os.makedirs(f"{R}/screen_L1", exist_ok=True)
LOCAL = os.environ.get("LOCAL_VOL")  # path of a local zarr root: render without S3 streaming
out_csv = f"{R}/screen_L1/{tag}_d{d}{'_local' if LOCAL else ''}.csv"
done = set()
if os.path.exists(out_csv):
    done = {int(r["winding"]) for r in csv.DictReader(open(out_csv))}

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

c = 33 + d
new = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    wr = csv.writer(f)
    if new:
        wr.writerow(["winding", "dir", "valid_px", "coarse_std", "band_std", "coherence"])
    for w in WINDINGS:
        if w in done:
            continue
        n = f"w{w:03d}"
        mesh = f"{run}/meshes/fitted_{tag}/{n}_{tag}"
        rz = f"{R}/screen_L1/{n}_{tag}.zarr"
        shutil.rmtree(rz, ignore_errors=True)
        vol_args = ["--volume", LOCAL] if LOCAL else ["--volume", f"{R}/ct_cache/volume.zarr", "--remote-url", URL]
        subprocess.run([VC, *vol_args, "--group-idx", "1", "--scale", "1.0",
                        "--segmentation", mesh, "--num-slices", "66", "--slice-step", "1", "--cache-gb", "6",
                        "--zarr-output", rz, "--timeout", "60"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tif = f"{R}/screen_L1/{n}_{tag}_d{d}.tif"
        subprocess.run(["D:/divesh-github/inkenv/Scripts/python.exe", "-m", "vesuvius.ink_detection.inference.infer", rz, CK, tif,
                        "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4", "--direction", "both",
                        "--layer-start", str(c - 8), "--layer-end", str(c + 9)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for suffix, name in (("", "forward"), ("_reverse", "reverse")):
            p = tif.replace(".tif", f"{suffix}.tif")
            if not os.path.exists(p):
                wr.writerow([w, name, "", "", "", ""]); continue
            s = stats(tifffile.imread(p))
            wr.writerow([w, name] + ([s["valid_px"], round(s["coarse_std"], 2), round(s["band_std"], 2), round(s["coherence"], 3)] if s else ["", "", "", ""]))
        f.flush()
        shutil.rmtree(rz, ignore_errors=True)
        for suffix in ("", "_reverse"):
            p = tif.replace(".tif", f"{suffix}.tif")
            if os.path.exists(p):
                os.replace(p, p.replace(R + "/screen_L1/", R + "/screen_L1/keep_")) if False else None
        print("done", n, flush=True)
