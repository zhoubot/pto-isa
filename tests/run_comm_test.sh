#!/usr/bin/env bash
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ST_DIR="${ROOT_DIR}/tests/npu/a2a3/comm/st/testcase"

if [[ ! -d "${ST_DIR}" ]]; then
  echo "[ERROR] testcase dir not found: ${ST_DIR}" >&2
  exit 1
fi

echo "[INFO] Running all comm ST testcases under ${ST_DIR}"

declare -a tests=()
while IFS= read -r -d '' dir; do
  name="$(basename "${dir}")"
  tests+=("${name}")
done < <(find "${ST_DIR}" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)

if [[ "${#tests[@]}" -eq 0 ]]; then
  echo "[ERROR] No testcase directories found under ${ST_DIR}" >&2
  exit 1
fi

fail_count=0
for t in "${tests[@]}"; do
  echo "============================================================"
  echo "[INFO] Running testcase: ${t}"
  echo "============================================================"
  if ! python3 "${ROOT_DIR}/tests/script/run_st.py" -r npu -v a3 -t "comm/${t}"; then
    echo "[ERROR] Testcase failed: ${t}" >&2
    fail_count=$((fail_count + 1))
  fi
done

echo "============================================================"
if [[ "${fail_count}" -eq 0 ]]; then
  echo "[INFO] All comm ST testcases passed."
  exit 0
else
  echo "[ERROR] ${fail_count} testcase(s) failed."
  exit 1
fi
