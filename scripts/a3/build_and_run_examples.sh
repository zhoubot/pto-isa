#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo "ASCEND_HOME_PATH is not set. On this machine it is typically:"
  echo "  export ASCEND_HOME_PATH=/home/zhouruoyu/Ascend/ascend-toolkit/latest"
  exit 2
fi

if [[ ! -d "${ASCEND_HOME_PATH}" ]]; then
  echo "ASCEND_HOME_PATH does not exist: ${ASCEND_HOME_PATH}"
  exit 2
fi

BUILD_DIR="${ROOT}/build/a3"
LIB_DIR="${BUILD_DIR}/kernels"
mkdir -p "${LIB_DIR}"

PTO_RUN_MODE="${PTO_RUN_MODE:-npu}"  # npu|sim
if [[ -z "${PTO_SOC_VERSION:-}" ]]; then
  if [[ "${PTO_RUN_MODE}" == "sim" ]]; then
    PTO_SOC_VERSION="Ascend910B1"
  else
    PTO_SOC_VERSION="Ascend910B"
  fi
fi
SIM_LIBDIR="${ASCEND_HOME_PATH}/tools/simulator/${PTO_SOC_VERSION}/lib"
SIM_ALT_LIBDIR="${ASCEND_HOME_PATH}/aarch64-linux/simulator/${PTO_SOC_VERSION}/lib"
SIM_LEGACY_LIBDIR="${ASCEND_HOME_PATH}/simulator/${PTO_SOC_VERSION}/lib"
SIM_TOOLS_LIBDIR="${ASCEND_HOME_PATH}/tools/simulator/${PTO_SOC_VERSION}/lib"

PTO_AICORE_ARCH="${PTO_AICORE_ARCH:-dav-c220-vec}"
PTO_CCE_COMPILER="${PTO_CCE_COMPILER:-ccec}"
PTO_CCE_MLLVM_FLAGS=(
  "-mllvm" "-cce-aicore-stack-size=0x8000"
  "-mllvm" "-cce-aicore-function-stack-size=0x8000"
  "-mllvm" "-cce-aicore-record-overflow=true"
  "-mllvm" "-cce-aicore-addr-transform"
  "-mllvm" "-cce-aicore-dcci-insert-for-scalar=false"
)
PTO_CCE_EXTRA_FLAGS="${PTO_CCE_EXTRA_FLAGS:-}"

if [[ "${PTO_RUN_MODE}" != "npu" && "${PTO_RUN_MODE}" != "sim" ]]; then
  echo "Unsupported PTO_RUN_MODE=${PTO_RUN_MODE} (expected npu|sim)"
  exit 2
fi
if [[ "${PTO_RUN_MODE}" == "sim" ]]; then
  SIM_LIBDIRS=()
  if [[ -d "${SIM_ALT_LIBDIR}" ]]; then
    SIM_LIBDIRS+=("${SIM_ALT_LIBDIR}")
  fi
  if [[ -d "${SIM_LEGACY_LIBDIR}" ]]; then
    SIM_LIBDIRS+=("${SIM_LEGACY_LIBDIR}")
  fi
  if [[ -d "${SIM_LIBDIR}" ]]; then
    SIM_LIBDIRS+=("${SIM_LIBDIR}")
  fi
else
  SIM_LIBDIRS=("${SIM_LIBDIR}")
fi
if [[ "${PTO_RUN_MODE}" == "sim" && "${#SIM_LIBDIRS[@]}" -eq 0 ]]; then
  echo "Simulator lib dir not found for SOC ${PTO_SOC_VERSION}."
  echo "Try setting PTO_SOC_VERSION (e.g. Ascend910B1 / Ascend910B)."
  exit 2
fi

echo "==> [1/4] Generate example outputs (Python -> .pto/.cpp)"
if [[ "${PTO_SKIP_GENERATE:-0}" != "1" ]]; then
  examples=(
    pto_isa_sinh.py
    pto_fused_softmax.py
    pto_aten_ir_primitives.py
    pto_torch_tensor.py
    pto_torch_functional.py
    pto_torch_nn_operators.py
    pto_torch_flexattention.py
    pto_llama7B_dynamic.py
  )
  if [[ "${PTO_SKIP_LLAMA:-0}" == "1" ]]; then
    examples=(
      pto_isa_sinh.py
      pto_fused_softmax.py
      pto_aten_ir_primitives.py
      pto_torch_tensor.py
      pto_torch_functional.py
      pto_torch_nn_operators.py
      pto_torch_flexattention.py
    )
  fi

  for ex in "${examples[@]}"; do
    echo "  - python examples/${ex}"
    python "${ROOT}/examples/${ex}"
  done
else
  echo "  - PTO_SKIP_GENERATE=1: skipping Python generation"
fi

echo "==> [2/4] Build NPU runner"
runner_out="${BUILD_DIR}/pto_npu_runner"
runner_runtime_lib="-lruntime"
runner_extra_rpath=()
runner_extra_ldpath=""
if [[ "${PTO_RUN_MODE}" == "sim" ]]; then
  runner_out="${BUILD_DIR}/pto_npu_runner_sim"
  runner_runtime_lib="-lruntime_camodel"
  for libdir in "${SIM_LIBDIRS[@]}"; do
    runner_extra_rpath+=("-Wl,-rpath,${libdir}")
  done
  runner_extra_ldpath="$(IFS=:; echo "${SIM_LIBDIRS[*]}:")"
fi

ccec -xc++ -O2 -std=c++17 \
  "${ROOT}/scripts/a3/pto_npu_runner.cpp" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/pkg_inc" \
  -L"${ASCEND_HOME_PATH}/lib64" \
  $([[ "${PTO_RUN_MODE}" == "sim" ]] && printf "%s " "${SIM_LIBDIRS[@]/#/-L}") \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  "${runner_extra_rpath[@]}" \
  -lascendcl ${runner_runtime_lib} -lnnopbase -ltiling_api -lc_sec -lplatform -ldl -lpthread -lm -lstdc++ \
  -o "${runner_out}"

echo "==> [3/4] Compile generated kernels to .so (CCE mode, ${PTO_AICORE_ARCH})"
if [[ ! -d "${ROOT}/examples/output_ascend_a2a3" ]]; then
  echo "Missing ${ROOT}/examples/output_ascend_a2a3; run generation first."
  exit 2
fi

cpp_glob="${PTO_CPP_GLOB:-*.cpp}"
search_dirs=("${ROOT}/examples/output_ascend_a2a3")
if [[ -n "${PTO_SUBDIRS:-}" ]]; then
  IFS=',' read -r -a subs <<< "${PTO_SUBDIRS}"
  search_dirs=()
  for s in "${subs[@]}"; do
    s="${s#/}"
    search_dirs+=("${ROOT}/examples/output_ascend_a2a3/${s}")
  done
fi

mapfile -t cpp_files < <(find "${search_dirs[@]}" -type f -name "${cpp_glob}" 2>/dev/null | sort)
if [[ "${#cpp_files[@]}" -eq 0 ]]; then
  echo "No matching .cpp files found."
  echo "  - base: ${ROOT}/examples/output_ascend_a2a3"
  echo "  - PTO_SUBDIRS=${PTO_SUBDIRS:-<unset>}"
  echo "  - PTO_CPP_GLOB=${cpp_glob}"
  exit 2
fi

compile_failures=0
built_sos=()
for cpp in "${cpp_files[@]}"; do
  rel="${cpp#${ROOT}/examples/output_ascend_a2a3/}"
  subdir="$(dirname "${rel}")"
  base="$(basename "${cpp}" .cpp)"
  out_dir="${LIB_DIR}/${subdir}"
  mkdir -p "${out_dir}"
  so="${out_dir}/${base}.so"
  obj="${out_dir}/${base}.o"

  echo "  - ${rel} -> ${so#${ROOT}/}"
  if ! "${PTO_CCE_COMPILER}" -xcce --cce-aicore-arch="${PTO_AICORE_ARCH}" -O2 -std=c++17 \
    -Xhost-start -Xhost-end \
    "${PTO_CCE_MLLVM_FLAGS[@]}" ${PTO_CCE_EXTRA_FLAGS} \
    -DMEMORY_BASE -DPTO_NPU_SMOKE_RUNNER \
    -fPIC -c \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -I"${ROOT}/include" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_HOME_PATH}/pkg_inc" \
    -I"${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I"${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    "${cpp}" \
    -o "${obj}"; then
    echo "    [compile-failed] ${rel}"
    compile_failures=$((compile_failures + 1))
    continue
  fi

  if ! "${PTO_CCE_COMPILER}" -shared "${obj}" -o "${so}" --cce-fatobj-link; then
    echo "    [link-failed] ${rel}"
    compile_failures=$((compile_failures + 1))
    continue
  fi
  built_sos+=("${so}")
done

echo "==> [4/4] Run kernels on NPU (ACL)"
if [[ "${#built_sos[@]}" -eq 0 ]]; then
  echo "No kernels built; nothing to run."
  exit 1
fi
if [[ "${PTO_RUN_MODE}" == "sim" ]]; then
  if [[ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    # shellcheck disable=SC1090
    source "${ASCEND_HOME_PATH}/bin/setenv.bash"
  fi
  LD_LIBRARY_PATH="${runner_extra_ldpath}${LD_LIBRARY_PATH:-}" \
    "${runner_out}" "${built_sos[@]}"
else
  LD_LIBRARY_PATH="${runner_extra_ldpath}${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
    "${runner_out}" "${built_sos[@]}"
fi

if [[ "${compile_failures}" -ne 0 ]]; then
  echo "Kernel compilation failures: ${compile_failures}"
  exit 1
fi
