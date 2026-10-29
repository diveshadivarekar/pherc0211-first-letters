"""Markdown table of the edge-band ink test (results/paris4_edge_band/*.json, written by ink_edge_band.py)."""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paris4_edge_band")
print("| segment | B (px) | points | band AUC | null 5-95% | p (2-sided) | high-passed AUC | null 5-95% | p (2-sided) |")
print("|---|---|---|---|---|---|---|---|---|")
for f in sorted(d.glob("*.json")):
    r = json.loads(f.read_text())
    for b, row in r["bands"].items():
        cells = []
        for k in ("band", "band_hp"):
            x = row[k]
            cells += [f"{x['observed']:.3f}", f"{x['null_p05']:.2f}-{x['null_p95']:.2f}", f"{x['p_two_sided']:.2f}"] if "null_p05" in x else ["n/a"] * 3
        print(f"| `{f.stem}` | {float(b):g} | {row['points']:,} | " + " | ".join(cells) + " |")
