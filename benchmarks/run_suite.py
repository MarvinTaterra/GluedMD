"""Run every case in the GLUED benchmark suite, print a summary table.

Each case directory under ``benchmarks/`` ships its own ``benchmark.py``.
This runner invokes them as subprocesses (so each gets a fresh OpenMM
Context, fresh CUDA state, no inter-case interference) and aggregates
the per-case JSON records in ``benchmarks/results/``.

Cases that can't run on the current machine (no CUDA, missing CHARMM-GUI
data, missing large model, …) print a `[skip] CASE: reason` line and the
suite continues.

Usage::

    python benchmarks/run_suite.py
    python benchmarks/run_suite.py --only ad_vacuum,ad_water
    python benchmarks/run_suite.py --skip kor_deepcv_large
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys


CASES = [
    "ad_vacuum",
    "ad_water",
    "kor_a100",
    "kor_deepcv_small",
    "kor_deepcv_large",
    "kv12_s4",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="",
                   help="Comma-separated case names — run only these.")
    p.add_argument("--skip", default="",
                   help="Comma-separated case names — skip these.")
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
    return p.parse_args()


def run_case(case: str, output_dir: str) -> dict | None:
    """Run one case as a subprocess. Returns the JSON record or None."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), case, "benchmark.py")
    if not os.path.exists(script):
        print(f"[skip] {case}: no {script}")
        return None
    print(f"\n──── {case} ────────────────────────────────────────────")
    proc = subprocess.run(
        [sys.executable, script, "--output-dir", output_dir],
        capture_output=False)
    if proc.returncode != 0:
        print(f"[skip] {case}: exit code {proc.returncode}")
        return None
    record_path = os.path.join(output_dir, f"{case}.json")
    if os.path.exists(record_path):
        with open(record_path) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    only = set(s for s in args.only.split(",") if s)
    skip = set(s for s in args.skip.split(",") if s)
    cases = [c for c in CASES
             if (not only or c in only)
             and c not in skip]

    print(f"GLUED benchmark suite — {len(cases)} case(s) selected")
    print(f"Output: {args.output_dir}")
    print(f"Started: {datetime.datetime.now().isoformat(timespec='seconds')}")

    records = []
    for case in cases:
        rec = run_case(case, args.output_dir)
        if rec is not None:
            records.append(rec)

    # Summary table
    print(f"\n{'='*78}\nSummary\n{'='*78}")
    print(f"{'case':<22}{'atoms':>10}{'steps/s':>12}{'ns/day':>10}  CV / bias")
    print(f"{'-'*22}{'-'*10}{'-'*12}{'-'*10}  {'-'*30}")
    for r in records:
        print(f"{r['benchmark']:<22}{r.get('n_atoms', '–'):>10}"
              f"{r['steps_per_sec']:>12.1f}{r['ns_per_day']:>10.1f}"
              f"  {r.get('cv_kind', '?')} / {r.get('bias_kind', '?')}")

    # Aggregate file with timestamp
    os.makedirs(args.output_dir, exist_ok=True)
    agg_path = os.path.join(args.output_dir, "suite.json")
    with open(agg_path, "w") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(),
                   "records": records}, f, indent=2)
    print(f"\nAggregate: {agg_path}")


if __name__ == "__main__":
    main()
