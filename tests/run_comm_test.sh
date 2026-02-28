#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# ============================================================================
# Usage
# ============================================================================
usage() {
  cat <<EOF
Usage: $(basename "$0") [-n NPU_COUNT] [-t TESTCASE]

Options:
  -n NPU_COUNT   Number of NPUs (devices) available: 2, 4, or 8 (default: 8)
                 Only test cases requiring <= NPU_COUNT ranks will run.
  -t TESTCASE    Run only the specified testcase (e.g. tput, treduce).
                 Can be specified multiple times. Default: run all.
  -h             Show this help message.

Examples:
  $(basename "$0")              # Run all tests with 8 NPUs
  $(basename "$0") -n 2         # Run only 2-rank tests
  $(basename "$0") -n 4 -t tput # Run tput tests requiring <= 4 ranks
EOF
  exit 0
}

# ============================================================================
# Parse arguments
# ============================================================================
NPU_COUNT=8
declare -a SELECTED_TESTS=()

while getopts "n:t:h" opt; do
  case "$opt" in
    n) NPU_COUNT="$OPTARG" ;;
    t) SELECTED_TESTS+=("$OPTARG") ;;
    h) usage ;;
    *) usage ;;
  esac
done

if [[ "$NPU_COUNT" != 2 && "$NPU_COUNT" != 4 && "$NPU_COUNT" != 8 ]]; then
  echo "[ERROR] -n must be 2, 4, or 8 (got: ${NPU_COUNT})" >&2
  exit 1
fi

# ============================================================================
# Build gtest filter per testcase based on NPU_COUNT.
#
# Tests that don't follow the *_NRanks / *_Nranks naming convention but
# require >2 ranks are listed explicitly below.
# ============================================================================
get_gtest_filter() {
  local test_name="$1"
  local filter="*"

  # Exclude 8-rank tests when NPU_COUNT < 8
  if (( NPU_COUNT < 8 )); then
    filter="${filter}:-*8Ranks*:-*8ranks*"
    case "$test_name" in
      tput) filter="${filter}:-TPut.Vec_Uint8Small" ;;
    esac
  fi

  # Exclude 4-rank tests when NPU_COUNT < 4
  if (( NPU_COUNT < 4 )); then
    filter="${filter}:-*4Ranks*:-*4ranks*"
    case "$test_name" in
      tput)       filter="${filter}:-TPut.Vec_FloatSmall:-TPut.AtomicAdd_Int32" ;;
      treduce)    filter="${filter}:-TReduce.FloatSmall_Sum" ;;
      tbroadcast) filter="${filter}:-TBroadCast.FloatSmallRoot0" ;;
      tgather)    filter="${filter}:-TGather.FloatSmall" ;;
      tscatter)   filter="${filter}:-TScatter.FloatSmall" ;;
    esac
  fi

  echo "$filter"
}

# ============================================================================
# Discover testcases
# ============================================================================
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ST_DIR="${ROOT_DIR}/tests/npu/a2a3/comm/st/testcase"

if [[ ! -d "${ST_DIR}" ]]; then
  echo "[ERROR] testcase dir not found: ${ST_DIR}" >&2
  exit 1
fi

declare -a tests=()
if [[ "${#SELECTED_TESTS[@]}" -gt 0 ]]; then
  tests=("${SELECTED_TESTS[@]}")
else
  while IFS= read -r -d '' dir; do
    tests+=("$(basename "${dir}")")
  done < <(find "${ST_DIR}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
fi

if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "[ERROR] No testcase directories found under ${ST_DIR}" >&2
  exit 1
fi

echo "[INFO] NPU_COUNT=${NPU_COUNT}, running ${#tests[@]} testcase(s): ${tests[*]}"

# ============================================================================
# Run
# ============================================================================
fail_count=0
for t in "${tests[@]}"; do
  gtest_filter="$(get_gtest_filter "$t")"

  echo "============================================================"
  echo "[INFO] Running testcase: ${t}  (GTEST_FILTER=${gtest_filter})"
  echo "============================================================"

  if ! GTEST_FILTER="${gtest_filter}" \
       python3 "${ROOT_DIR}/tests/script/run_st.py" -r npu -v a3 -t "comm/${t}"; then
    echo "[ERROR] Testcase failed: ${t}" >&2
    fail_count=$((fail_count + 1))
  fi
done

echo "============================================================"
if [[ "${fail_count}" -eq 0 ]]; then
  echo "[INFO] All ${#tests[@]} comm ST testcase(s) passed (NPU_COUNT=${NPU_COUNT})."
  exit 0
else
  echo "[ERROR] ${fail_count}/${#tests[@]} testcase(s) failed."
  exit 1
fi
