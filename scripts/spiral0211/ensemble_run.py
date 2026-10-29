"""Run the winner's 4 ink models (ink_9um seed 42/43, dense_native, Hecate 9.6um) on a 66-layer render.

Layout follows the PHerc 343 winner's run_ink.sh: villa models use layers s = 25+d .. s+16 (17 layers), direction both;
Hecate gets 16 planes resampled to 9.6 um, centre z = 32.5 + d, forward and --reverse. Outputs (uint8):
  <out>/<label>__<model>__d<d>.tif|png and ..._reverse.tif|png   (forward = blind side, _reverse = reading side per winner)
Run with D:/divesh-github/inkenv/Scripts/python.exe.
Usage: ensemble_run.py <render.zarr> <out_dir> <label> <native_um> <d1,d2,...> [models comma list]
"""
import os, subprocess, sys
import numpy as np, zarr, cv2, tifffile

render, out, label, native_um = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
depths = [int(x) for x in sys.argv[5].split(",")]
models = sys.argv[6].split(",") if len(sys.argv) > 6 else ["seed42", "seed43", "dnative", "hecate"]
os.makedirs(out, exist_ok=True)
PY = "D:/divesh-github/inkenv/Scripts/python.exe"
CK = {"seed42": "D:/divesh-github/ink9um/hybrid_3d2d-seed42/step-075000.pth",
      "seed43": "D:/divesh-github/ink_models/seed43/step-075000.pth",
      "dnative": "D:/divesh-github/ink_models/dense_native/dense_native-016000.rebuilt.pth"}
HEC = "D:/divesh-github/ink_models/hecate/hecate_9.6um.pth"
HEC_PY = "D:/divesh-github/ink_models/hecate/hecate.py"

def villa(model, d):
    base = f"{out}/{label}__{model}__d{d}"
    if os.path.exists(base + ".tif") and os.path.exists(base + "_reverse.tif"):
        return
    s = 25 + d
    subprocess.run([PY, "-m", "vesuvius.ink_detection.inference.infer", render, CK[model], base + ".tif", "--overlap", "0.5",
                    "--blend-mode", "hann", "--batch-size", "4", "--direction", "both", "--layer-start", str(s), "--layer-end", str(s + 17)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def hecate(d):
    base = f"{out}/{label}__hecate__d{d}"
    if os.path.exists(base + ".png") and os.path.exists(base + "_reverse.png"):
        return
    vol = zarr.open(render, mode="r")
    arr = vol["0"] if "0" in vol else vol
    n, H, W = arr.shape
    zc = (n - 1) / 2 + d                      # 32.5 + d for 66 layers
    step = 9.6 / native_um                    # source layers per 9.6 um plane
    zs = zc + (np.arange(16) - 7.5) * step
    lo, hi = int(np.floor(zs.min())), int(np.ceil(zs.max()))
    block = np.asarray(arr[lo:hi + 1]).astype(np.float32)
    planes = []
    for z in zs:
        i = z - lo; i0 = int(np.floor(i)); i1 = min(i0 + 1, block.shape[0] - 1); f = i - i0
        planes.append((1 - f) * block[i0] + f * block[i1])
    f_xy = native_um / 9.6
    h2, w2 = int(round(H * f_xy)), int(round(W * f_xy))
    prep = np.stack([cv2.resize(p, (w2, h2), interpolation=cv2.INTER_AREA if f_xy < 1 else cv2.INTER_LINEAR) for p in planes])
    prep = np.clip(np.rint(prep), 0, 255).astype(np.uint8)
    npy = f"{out}/_hecate_prep_{label}_d{d}.npy"
    np.save(npy, prep)
    for rev in (False, True):
        o = base + ("_reverse" if rev else "") + ".png"
        if os.path.exists(o): os.remove(o)
        cmd = [PY, HEC_PY, "--checkpoint", HEC, "--input", npy, "--spacing-um", "9.6", "--output", o, "--batch-size", "4"]
        if rev: cmd.append("--reverse")
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(o):  # back to render pixel grid so all models share one frame
            m = cv2.imread(o, cv2.IMREAD_UNCHANGED)
            cv2.imwrite(o.replace(".png", "_native.tif"), cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR))
    os.remove(npy)

for d in depths:
    for m in models:
        (hecate(d) if m == "hecate" else villa(m, d))
        print("done", m, d, flush=True)
print("ENSEMBLE_RUN_DONE", flush=True)
