"""Run the full M1-M5 pipeline against one phantom, headlessly (no GUI).

Thin CLI wrapper around src.m6_gui.orchestrator.PipelineOrchestrator — the
same orchestration logic the GUI (M6) drives — for scripting/CI use.

Usage:
    python scripts/run_end_to_end.py --phantom data/phantoms/phantom_001.h5 --out data/processed/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.m6_gui.orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phantom", required=True, help="Phantom HDF5 (from generate_phantom)")
    parser.add_argument("--out", required=True, help="Working/output directory for every phase")
    args = parser.parse_args()

    def log(phase: str, message: str) -> None:
        print(f"[{phase}] {message}")

    orchestrator = PipelineOrchestrator(args.out)
    results = orchestrator.run_all(args.phantom, callback=log)

    print("\n--- Summary ---")
    for r in results:
        status = "OK" if r.ok else "FAILED"
        print(f"{r.name}: {status} — {r.message}")

    if not all(r.ok for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
