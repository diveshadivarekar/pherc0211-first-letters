"""List the volumes and published representations of Vesuvius Challenge scrolls (S3 listing only, nothing downloaded).

Usage: python scripts/survey_scrolls.py PHerc0125 PHerc0211 ... [--json out.json]
Volume folder names encode scan id, voxel size, beam energy and (when masked) the processing, e.g.
``20250821151803-9.362um-1.2m-113keV-masked.zarr``.
"""
import argparse, json, re
import s3fs

BUCKET = "vesuvius-challenge-open-data"
p = argparse.ArgumentParser(); p.add_argument("scrolls", nargs="+"); p.add_argument("--json")
a = p.parse_args()
fs = s3fs.S3FileSystem(anon=True)
top = {x.split("/")[-1] for x in fs.ls(BUCKET) if fs.isdir(x)}
out = {}
for s in a.scrolls:
    if s not in top:
        print(f"{s}: no such prefix in bucket"); out[s] = None; continue
    rec = {"volumes": [], "representations": {}}
    for v in fs.ls(f"{BUCKET}/{s}/volumes/"):
        name = v.split("/")[-1]
        m = re.search(r"(\d{14})-([\d.]+)um-([\d.]+)m-(\d+)keV", name)
        rec["volumes"].append({"name": name, "scan": m[1] if m else None, "um": float(m[2]) if m else None, "keV": int(m[4]) if m else None})
    try:
        for r in fs.ls(f"{BUCKET}/{s}/representations/"):
            kind = r.split("/")[-1]
            rec["representations"][kind] = [x.split("/")[-1] for x in fs.ls(r)][:12]
    except FileNotFoundError:
        pass
    out[s] = rec
    vols = "; ".join(f"{v['um']}um/{v['keV']}keV" for v in rec["volumes"])
    reps = {k: len(v) for k, v in rec["representations"].items()}
    print(f"{s}: {len(rec['volumes'])} vol [{vols}] reps {reps}", flush=True)
if a.json:
    open(a.json, "w").write(json.dumps(out, indent=1))
