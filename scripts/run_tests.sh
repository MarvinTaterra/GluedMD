#!/usr/bin/env bash
# scripts/run_tests.sh — full local test + perf suite
#
# Usage:
#   ./scripts/run_tests.sh              # build + all tests + perf
#   ./scripts/run_tests.sh --no-build   # skip cmake (use existing build/)
#   ./scripts/run_tests.sh --no-perf    # skip benchmark comparison
#   ./scripts/run_tests.sh --no-parity  # skip PLUMED parity tests
#   ./scripts/run_tests.sh --update-baselines  # write new baseline JSONs
#
# Requires: conda env 'openmm_env' active (or OPENMM_DIR set manually).

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
pass()  { echo -e "${GREEN}PASS${NC}  $*"; }
warn()  { echo -e "${YELLOW}WARN${NC}  $*"; }
fail()  { echo -e "${RED}FAIL${NC}  $*"; }
header(){ echo -e "\n${YELLOW}══ $* ══${NC}"; }

# ── Defaults ─────────────────────────────────────────────────────────────────
DO_BUILD=1
DO_PERF=1
DO_PARITY=1
UPDATE_BASELINES=0
THRESHOLD=0.90   # fail if a benchmark drops below 90% of baseline

for arg in "$@"; do
  case "$arg" in
    --no-build)          DO_BUILD=0 ;;
    --no-perf)           DO_PERF=0 ;;
    --no-parity)         DO_PARITY=0 ;;
    --update-baselines)  UPDATE_BASELINES=1 ;;
    *) echo "Unknown option: $arg"; exit 1 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

RESULTS_DIR="$(mktemp -d /tmp/glued_bench_XXXXXX)"
FAILED=0

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ $DO_BUILD -eq 1 ]]; then
  header "Configure + Build"
  cmake -S . -B build -G Ninja \
    -DOPENMM_DIR="${OPENMM_DIR:-$CONDA_PREFIX}" \
    -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"$(nproc)"
  cmake --install build
  pass "Build complete"
fi

# ── Smoke test ────────────────────────────────────────────────────────────────
header "Smoke test"
if python tests/test_api_smoke.py; then
  pass "Smoke test"
else
  fail "Smoke test"; FAILED=1
fi

# ── Full unit test suite ───────────────────────────────────────────────────────
header "Unit tests  (tests/)"
if python -m pytest tests/ -q --tb=short; then
  pass "Unit tests"
else
  fail "Unit tests"; FAILED=1
fi

# ── PLUMED parity tests ───────────────────────────────────────────────────────
if [[ $DO_PARITY -eq 1 ]]; then
  header "PLUMED parity tests  (tests_plumed_parity/)"
  if python -c "import subprocess; subprocess.run(['plumed','--version'],check=True,capture_output=True)" 2>/dev/null; then
    if python -m pytest tests_plumed_parity/ -q --tb=short; then
      pass "PLUMED parity"
    else
      fail "PLUMED parity"; FAILED=1
    fi
  else
    warn "plumed not found — parity tests skipped (install via: conda install -c conda-forge plumed)"
  fi
fi

# ── Benchmarks ────────────────────────────────────────────────────────────────
if [[ $DO_PERF -eq 1 ]]; then
  header "Benchmarks"

  run_bench() {
    local script=$1 out=$2 label=$3
    if python "benchmarks/$script" --out "$out" 2>&1; then
      pass "$label"
    else
      warn "$label skipped (missing dependency or no GPU)"
    fi
  }

  run_bench bench_adp.py          "$RESULTS_DIR/bench_adp.json"          "ADP vacuum"
  run_bench bench_solvated_adp.py "$RESULTS_DIR/bench_solvated.json"     "ADP solvated"
  run_bench bench_multiwalker.py  "$RESULTS_DIR/bench_multiwalker.json"  "Multi-walker"
  run_bench bench_pytorch_cv.py   "$RESULTS_DIR/bench_pytorch.json"      "PyTorch CV"

  if [[ $UPDATE_BASELINES -eq 1 ]]; then
    header "Updating baselines"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null \
               | head -1 | tr ' ' '_' || echo "unknown_gpu")
    DEST="benchmarks/baselines/${GPU_NAME}.json"
    # Merge all result files into one baseline JSON
    python - <<EOF
import json, glob, pathlib, sys

files = glob.glob("$RESULTS_DIR/*.json")
merged = {}
for f in files:
    merged.update(json.load(open(f)))

out = pathlib.Path("$DEST")
json.dump(merged, open(out, "w"), indent=2)
print(f"Wrote baseline → {out}  ({len(merged)} entries)")
EOF
    pass "Baselines updated → $DEST"
  else
    header "Regression check  (threshold: $(echo "$THRESHOLD * 100" | bc -l | xargs printf '%.0f')%)"
    if python benchmarks/check_regression.py \
        "$RESULTS_DIR/" benchmarks/baselines/ \
        --threshold "$THRESHOLD"; then
      pass "No regressions"
    else
      fail "Performance regression detected"; FAILED=1
    fi
  fi

  echo ""
  echo "Raw results: $RESULTS_DIR/"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [[ $FAILED -eq 0 ]]; then
  echo -e "${GREEN}All checks passed.${NC}"
else
  echo -e "${RED}One or more checks failed — see output above.${NC}"
  exit 1
fi
