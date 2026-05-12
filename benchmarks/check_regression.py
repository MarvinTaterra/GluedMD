"""CI regression check: compare fresh benchmark results to stored baselines.

Usage:
    python benchmarks/check_regression.py <results_dir> <baselines_dir> [--threshold 0.90]

Each JSON file in results_dir is matched to a baseline in baselines_dir by
the GPU name embedded in the result file ("gpu_name" key).  Exits non-zero
if any variant regresses below threshold × baseline steps/s.
"""

import argparse
import json
import sys
from pathlib import Path

from utils import compare_to_baseline, load_baseline


def _gpu_key(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("baselines_dir")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Minimum ratio of current/baseline sps (default 0.90 = 10%% regression)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    baselines_dir = Path(args.baselines_dir)
    threshold = args.threshold

    result_files = list(results_dir.glob("*.json"))
    if not result_files:
        print("No result files found — nothing to compare.")
        return 0

    all_regressions = []

    for rf in sorted(result_files):
        with open(rf) as fh:
            result = json.load(fh)

        gpu_name = result.get("gpu_name", "unknown")
        gpu_key = _gpu_key(gpu_name)

        # Find matching baseline (GPU name embedded in filename)
        candidates = list(baselines_dir.glob(f"{gpu_key}*.json"))
        # Strip multiwalker/pytorch/solvated suffixes to get the primary baseline
        primary = [c for c in candidates
                   if not any(s in c.stem for s in ("multiwalker", "pytorch", "solvated"))]

        if not primary:
            print(f"  {rf.name}: no baseline for GPU '{gpu_name}' — skipping")
            continue

        baseline = load_baseline(str(primary[0]))
        if baseline is None:
            print(f"  {rf.name}: baseline unreadable — skipping")
            continue

        regressions = compare_to_baseline(result, baseline, threshold=threshold)
        label = rf.stem

        if regressions:
            for r in regressions:
                pct = r["ratio"] * 100
                print(
                    f"  REGRESSION [{label}] {r['variant']}: "
                    f"{r['current_sps']:.0f} sps vs baseline {r['baseline_sps']:.0f} sps "
                    f"({pct:.1f}% — below {threshold*100:.0f}% threshold)"
                )
            all_regressions.extend(regressions)
        else:
            print(f"  OK [{label}]: no regressions vs {primary[0].name}")

    if all_regressions:
        print(f"\n{len(all_regressions)} regression(s) detected. Failing.")
        return 1

    print("\nAll benchmarks within threshold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
