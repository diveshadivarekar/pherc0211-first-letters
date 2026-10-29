"""Markdown tables from the per-segment experiment records (ink_segment_experiment.py) and spatial-null records (ink_spatial_null.py).

Usage (from the repo root): python scripts/probes/summarize_multi_segment.py
"""

from __future__ import annotations

import json
from pathlib import Path

MULTI = Path("data/processed/paris4_multi")
W00_NULLS = {"split": Path("data/processed/paris4/null_w00_split.json"), "baseline": Path("data/processed/paris4/null_w00_baseline.json")}
SEGMENTS = ["w00", "w01", "w02", "w03", "w05"]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    print("| segment | tile labels | ink | share reached by pipeline (+-8) / official | common pixels | all-ink F0.5 | official+refine AUC | pipeline +-8 AUC | pipeline +-12 AUC |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in SEGMENTS:
        r = load(MULTI / s / f"{s}.json")
        if r is None:
            print(f"| `{s}` | pending | | | | | | | |")
            continue
        c = r["common"]
        a12 = c["pipeline +-12"].get("auc")
        print(f"| `{s}` | {r['label_pixels'] / 1e6:.2f}M | {100 * r['ink_fraction']:.0f}% | {100 * r['reached']['pipeline +-8']:.0f}% / {100 * r['reached']['gold + refine']:.0f}% | {r['common_pixels'] // 1000}k | "
              f"{c['gold + refine']['f0.5_all_ink']:.3f} | {c['gold + refine']['auc']:.3f} | {c['pipeline +-8']['auc']:.3f} | {'n/a' if a12 is None else format(a12, '.3f')} |")
    print()
    print("| segment | -6 | -3 | 0 | +3 | +6 | page, labels (ink) |")
    print("|---|---|---|---|---|---|---|")
    for s in SEGMENTS:
        r = load(MULTI / s / f"{s}.json")
        if r is None:
            continue
        d = r["depth_scan_gold"]
        a = d["auc_by_shift"]
        print(f"| `{s}` | " + " | ".join(f"{a[k]:.3f}" for k in ("-6", "-3", "0", "+3", "+6")) + f" | {d['page']}, {d['label_pixels'] // 1000}k ({100 * d['ink_fraction']:.0f}%) |")
    print()
    print("| segment | ink map scored | observed AUC | null mean | null SD | null 5-95% | null range | shifts | two-sided p |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = [("w00", "split pages", load(W00_NULLS["split"])), ("w00", "baseline pages", load(W00_NULLS["baseline"]))] + [(s, "split pages, refine +-8", load(MULTI / s / "null_s8.json")) for s in SEGMENTS[1:]]
    for s, what, n in rows:
        if n is None:
            print(f"| `{s}` | {what} | pending | | | | | | |")
            continue
        print(f"| `{s}` | {what} | {n['observed_auc']:.3f} | {n['null_mean']:.3f} | {n['null_sd']:.3f} | {n['null_p05']:.2f}-{n['null_p95']:.2f} | {n['null_min']:.2f}-{n['null_max']:.2f} | {n['null_n']} | {n['p_two_sided']:.2f} |")


if __name__ == "__main__":
    main()
