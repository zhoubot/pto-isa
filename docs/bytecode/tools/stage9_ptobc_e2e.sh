#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -z "$ROOT" ]]; then
  ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
fi

PTOBC_BIN=${PTOBC_BIN:-"$ROOT/build/ptobc/ptobc"}
SAMPLES_DIR=${SAMPLES_DIR:-"$ROOT/docs/bytecode/samples"}

if [[ ! -x "$PTOBC_BIN" ]]; then
  echo "ERROR: PTOBC_BIN not found/executable: $PTOBC_BIN" >&2
  echo "Build it first: ninja -C $ROOT/build/ptobc" >&2
  exit 2
fi

TMP=${TMPDIR:-/tmp}/ptobc_stage9_$$
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

PTOAS_BIN=${PTOAS_BIN:-"$ROOT/PTOAS/build/tools/ptoas/ptoas"}
RUN_PTOAS=${PTOBC_E2E_PTOAS:-0}

DEBUGINFO=${PTOBC_E2E_DEBUGINFO:-0}

run_encode() {
  local in="$1"
  local out="$2"
  if [[ "$DEBUGINFO" == "1" ]]; then
    PTOBC_EMIT_DEBUGINFO=1 "$PTOBC_BIN" encode "$in" -o "$out"
  else
    "$PTOBC_BIN" encode "$in" -o "$out"
  fi
}

run_decode() {
  local in="$1"
  local out="$2"
  if [[ "$DEBUGINFO" == "1" ]]; then
    PTOBC_PRINT_LOC=1 "$PTOBC_BIN" decode "$in" -o "$out"
  else
    "$PTOBC_BIN" decode "$in" -o "$out"
  fi
}

fail=0

echo "[stage9] PTOBC_BIN=$PTOBC_BIN"
echo "[stage9] SAMPLES_DIR=$SAMPLES_DIR"
echo "[stage9] TMP=$TMP"

for pto in "$SAMPLES_DIR"/*.pto; do
  base=$(basename "$pto" .pto)
  echo "[stage9] === $base ==="

  bc1="$TMP/$base.ptobc"
  pto2="$TMP/$base.dec.pto"
  bc2="$TMP/$base.re.ptobc"

  if ! run_encode "$pto" "$bc1"; then
    echo "[stage9] FAIL: encode $pto" >&2
    fail=1
    continue
  fi

  if ! run_decode "$bc1" "$pto2"; then
    echo "[stage9] FAIL: decode $bc1" >&2
    fail=1
    continue
  fi

  if ! run_encode "$pto2" "$bc2"; then
    echo "[stage9] FAIL: re-encode decoded .pto ($pto2)" >&2
    fail=1
    continue
  fi

  if [[ "$RUN_PTOAS" == "1" ]]; then
    if [[ ! -x "$PTOAS_BIN" ]]; then
      echo "[stage9] WARN: PTOAS_BIN not found/executable, skip ptoas parse: $PTOAS_BIN" >&2
    else
      # NOTE: ptoas runs a pass pipeline, so we only use it as an optional extra check.
      # Some synthetic samples (e.g. sync_stage7) may fail under ptoas pipeline even if MLIR parse succeeds.
      if [[ "$base" == "sync_stage7" ]]; then
        echo "[stage9] skip ptoas check for $base (known pipeline mismatch)"
      else
        if ! "$PTOAS_BIN" "$pto2" -o "$TMP/$base.cpp" >/dev/null 2>"$TMP/$base.ptoas.err"; then
          echo "[stage9] FAIL: ptoas parse $pto2" >&2
          tail -n 20 "$TMP/$base.ptoas.err" >&2 || true
          fail=1
          continue
        fi
      fi
    fi
  fi

done

# Extra: synthetic debug-location smoke test.
if [[ "$DEBUGINFO" == "1" ]]; then
  echo "[stage9] === debuginfo_smoke ==="
  loc_pto="$TMP/loc_test.pto"
  cat >"$loc_pto" <<'EOF'
module {
  func.func @f() {
    %c0_i32 = arith.constant 0 : i32 loc("x.cc":10:2)
    %c1_f32 = arith.constant 1.0 : f32 loc("x.cc":11:3)
    return loc("x.cc":12:1)
  } loc("x.cc":9:1)
} loc("x.cc":8:1)
EOF
  run_encode "$loc_pto" "$TMP/loc_test.ptobc"
  run_decode "$TMP/loc_test.ptobc" "$TMP/loc_test.dec.pto"
fi

if [[ "$fail" == "0" ]]; then
  echo "[stage9] PASS"
  exit 0
else
  echo "[stage9] FAIL" >&2
  exit 1
fi
