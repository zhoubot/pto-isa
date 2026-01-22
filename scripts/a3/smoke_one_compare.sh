#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"

MODE="${1:-sim}"        # sim|npu
SUBDIR="${2:-fused_softmax}"
CPP_GLOB="${3:-rowmax.cpp}"
PTO_SKIP_GENERATE="${PTO_SKIP_GENERATE:-1}"

if [[ "${MODE}" != "sim" && "${MODE}" != "npu" ]]; then
  echo "Usage: $0 [sim|npu] [subdir] [cpp_glob]"
  exit 2
fi

mkdir -p "${ROOT}/build/a3/logs" "${ROOT}/build/cpu" "${ROOT}/build/cpu/kernels"

CPU_RUNNER="${ROOT}/build/cpu/pto_cpu_runner"
g++ -O2 -std=c++17 \
  "${ROOT}/scripts/cpu/pto_cpu_runner.cpp" \
  -ldl \
  -o "${CPU_RUNNER}"

cpp_glob="${CPP_GLOB}"
c_glob="${CPP_GLOB//.cpp/.c}"

cpu_search_dirs=("${ROOT}/examples/output_arm64/${SUBDIR}")
mapfile -t c_files < <(find "${cpu_search_dirs[@]}" -type f -name "${c_glob}" 2>/dev/null | sort)
if [[ "${#c_files[@]}" -eq 0 ]]; then
  echo "No matching .c files found."
  echo "  - base: ${ROOT}/examples/output_arm64/${SUBDIR}"
  echo "  - C glob: ${c_glob}"
  exit 2
fi

cpu_sos=()
for c in "${c_files[@]}"; do
  rel="${c#${ROOT}/examples/output_arm64/}"
  subdir="$(dirname "${rel}")"
  base="$(basename "${c}" .c)"
  out_dir="${ROOT}/build/cpu/kernels/${subdir}"
  mkdir -p "${out_dir}"
  so="${out_dir}/${base}.so"
  echo "  - [cpu] ${rel} -> ${so#${ROOT}/}"
  gcc -shared -fPIC -O2 -std=c11 -DPTO_CPU_SMOKE_RUNNER \
    -I"${ROOT}" \
    -I"${ROOT}/include" \
    "${c}" \
    -o "${so}"
  cpu_sos+=("${so}")
done

cpu_log="${ROOT}/build/a3/logs/${SUBDIR//\//_}_${CPP_GLOB//\*/ALL}_${MODE}_cpu.log"
"${CPU_RUNNER}" "${cpu_sos[@]}" |& tee "${cpu_log}"

npu_log="${ROOT}/build/a3/logs/${SUBDIR//\//_}_${CPP_GLOB//\*/ALL}_${MODE}_npu.log"
PTO_RUN_MODE="${MODE}" \
PTO_SKIP_GENERATE="${PTO_SKIP_GENERATE}" \
PTO_SUBDIRS="${SUBDIR}" \
PTO_CPP_GLOB="${cpp_glob}" \
"${ROOT}/scripts/a3/build_and_run_examples.sh" |& tee "${npu_log}"

python "${ROOT}/scripts/a3/compare_checksums.py" "${cpu_log}" "${npu_log}"
echo "CPU log: ${cpu_log}"
echo "NPU log: ${npu_log}"
