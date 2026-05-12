"""Shared utilities for GLUED benchmarks."""

import json
import os
import subprocess
import time


class BenchmarkTimer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


def gpu_info():
    """Return list of {name, memory_mb} dicts from nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        rows = []
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split(",", 1)
            rows.append({"name": parts[0].strip(),
                         "memory_mb": int(parts[1].strip())})
        return rows or None
    except Exception:
        return None


def save_result(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"Saved: {path}")


def load_baseline(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def compare_to_baseline(result, baseline, threshold=0.90):
    """Return list of regression dicts for variants below threshold × baseline sps."""
    regressions = []
    for key, val in result.get("variants", {}).items():
        curr_sps = val.get("steps_per_second")
        if curr_sps is None:
            continue
        base_val = baseline.get("variants", {}).get(key, {})
        base_sps = base_val.get("steps_per_second")
        if base_sps is None:
            continue
        ratio = curr_sps / base_sps
        if ratio < threshold:
            regressions.append({
                "variant": key,
                "baseline_sps": base_sps,
                "current_sps": curr_sps,
                "ratio": ratio,
            })
    return regressions
