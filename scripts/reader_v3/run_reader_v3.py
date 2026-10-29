"""Reader v3 driver: one resumable, sequential pipeline per surface (render -> [sheet track] -> infer -> score -> row).

Replaces the reader_v2 chain*.sh scripts. Changes over v2:
  * one entry point; no hard-coded layer arithmetic (CENTER/offset/window derived from the render depth);
  * jobs run strictly one at a time and are skipped when their outputs exist (v2 crashed on Windows commit
    memory when jobs overlapped);
  * optional per-pixel sheet tracking (sheet A / B / B-edge, scripts/reader_v2/track_sheet.py) instead of a fixed offset;
  * scoring + component-shape stats (vertical / horizontal share, v2 `component_shape.py`) are done in the same run;
  * every run writes <out>/<name>/manifest.json (checkpoint sha256, args, git head) and appends to one CSV;
  * the positive control is re-checked first (--control); a failed control aborts the run so numbers are never
    reported against a broken setup.

Usage (run with D:/divesh-github/inkenv/Scripts/python.exe, from the repo root):
  run_reader_v3.py <render.zarr> <name> --offsets 0 7 -7 [--sheet mesh|A|B|Bedge] [--tta]
                   [--control <control_render.zarr> --control-name ctrl343 --control-offset 1]
"""
import argparse, csv, hashlib, importlib.util, json, os, subprocess, sys, time
import numpy as np, tifffile, zarr
from scipy import ndimage as ndi

PY = sys.executable
CKPT = "D:/divesh-github/reader-v2-weights/reader-v2-step040000.pth"
TEXTSCORE = "D:/divesh-github/reader-v2/scripts/textscore2.py"
TRACK = os.path.join(os.path.dirname(__file__), "..", "reader_v2", "track_sheet.py")
FLAG = 0.008                       # reader-v2 calibration: known text 0.010-0.032, blank/false 0.0003-0.0057
WIN = 17                           # model window (layers)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ts():
    spec = importlib.util.spec_from_file_location("textscore2", TEXTSCORE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def depth_of(render):
    z = zarr.open(render, mode="r")
    return (z["0"] if "0" in z else z).shape[0]


def infer(render, out_tif, centre, tta):
    """One sequential inference job; both directions; window of WIN layers centred on `centre`."""
    rev = out_tif[:-4] + "_reverse.tif"
    if os.path.exists(out_tif) and os.path.exists(rev):
        return
    s = centre - WIN // 2
    cmd = [PY, "-m", "vesuvius.ink_detection.inference.infer", render, CKPT, out_tif,
           "--overlap", "0.5", "--blend-mode", "hann", "--batch-size", "4", "--direction", "both",
           "--layer-start", str(s), "--layer-end", str(s + WIN)]
    if tta:
        cmd.append("--tta-mirror")
    with open(out_tif[:-4] + ".log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc or not os.path.exists(rev):
        raise RuntimeError(f"inference failed rc={rc}, see {out_tif[:-4]}.log")


def shape_share(p, ts_valid, thr=180):
    """Share of confident-ink component area that is vertically / horizontally elongated (control: 44% H / 26% V)."""
    lab, n = ndi.label((p >= thr) & ts_valid)
    if n == 0:
        return 0.0, 0.0
    objs = ndi.find_objects(lab)
    areas = ndi.sum(np.ones_like(lab), lab, range(1, n + 1))
    v = h = 0.0
    for sl, a in zip(objs, areas):
        hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if hh >= 1.5 * ww:
            v += a
        elif ww >= 1.5 * hh:
            h += a
    tot = float(areas.sum())
    return v / tot, h / tot


def score(ts, render, f, r):
    z = zarr.open(render, mode="r")
    arr = z["0"] if "0" in z else z
    pf, pr = tifffile.imread(f), tifffile.imread(r)
    h, w = pf.shape
    ct = np.asarray(arr[arr.shape[0] // 2, :h, :w])
    valid = (ct > 0) & (pf > 0)
    s_f, s_r = ts.score2(pf, pr, ct, valid), ts.score2(pr, pf, ct, valid)
    vf, hf = shape_share(pf, valid)
    vr, hr = shape_share(pr, valid)
    return dict(valid_px=int(valid.sum()), screen_fwd=s_f[180], screen_rev=s_r[180],
                vert_fwd=round(vf, 3), horiz_fwd=round(hf, 3), vert_rev=round(vr, 3), horiz_rev=round(hr, 3))


def run_surface(ts, render, name, offsets, sheet, tta, out):
    depth = depth_of(render)
    mesh_layer = depth // 2
    d = os.path.join(out, name); os.makedirs(d, exist_ok=True)
    src, centre0 = render, mesh_layer
    if sheet != "mesh":
        tr = os.path.join(d, f"track_{sheet}.zarr")
        if not os.path.exists(tr):
            cmd = [PY, TRACK, render, tr, "--sheet", sheet[0], "--half", str(WIN // 2 + 8)]
            if sheet == "Bedge":
                cmd.append("--edge")
            subprocess.run(cmd, check=True)
        src, centre0 = tr, depth_of(tr) // 2
    rows = []
    for off in offsets:
        centre = centre0 + off
        if not (WIN // 2 <= centre < depth_of(src) - WIN // 2):
            print(f"[{name}] offset {off} outside the stack, skipped"); continue
        t = os.path.join(d, f"{sheet}_d{off}.tif")
        t0 = time.time(); infer(src, t, centre, tta)
        row = dict(surface=name, sheet=sheet, offset=off, tta=int(tta), **score(ts, src, t, t[:-4] + "_reverse.tif"))
        row["flag"] = int(max(row["screen_fwd"], row["screen_rev"]) >= FLAG)
        rows.append(row); print(f"[{name}] {sheet} d={off} {time.time() - t0:.0f}s", row, flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("render"); ap.add_argument("name")
    ap.add_argument("--offsets", type=int, nargs="+", default=[0])
    ap.add_argument("--sheet", choices=["mesh", "A", "B", "Bedge"], default="mesh")
    ap.add_argument("--tta", action="store_true", help="mirror TTA (slower, usually smoother maps)")
    ap.add_argument("--out", default="data/reader_v3")
    ap.add_argument("--control"); ap.add_argument("--control-name", default="control")
    ap.add_argument("--control-offset", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ts = load_ts()
    rows = []
    if a.control:
        c = run_surface(ts, a.control, a.control_name, [a.control_offset], "mesh", a.tta, a.out)
        if not c or not c[0]["flag"]:
            sys.exit(f"CONTROL FAILED (flag threshold {FLAG}): {c}. Not running {a.name}.")
        rows += c
    rows += run_surface(ts, a.render, a.name, a.offsets, a.sheet, a.tta, a.out)
    manifest = dict(checkpoint=CKPT, ckpt_sha256=sha256(CKPT), args=vars(a), time=time.strftime("%F %T"),
                    villa_head=subprocess.run(["git", "-C", "D:/divesh-github/villa", "rev-parse", "HEAD"],
                                              capture_output=True, text=True).stdout.strip())
    with open(os.path.join(a.out, a.name, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    csv_path = os.path.join(a.out, "score_all.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if new:
            wr.writeheader()
        wr.writerows(rows)
    print("flagged:", [(r["surface"], r["sheet"], r["offset"]) for r in rows if r["flag"]] or "none")


if __name__ == "__main__":
    main()
