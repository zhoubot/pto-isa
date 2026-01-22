#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"

MODE="${1:-sim}"        # sim|npu
SUBDIR="${2:-fused_softmax}"
CPP_GLOB="${3:-rowmax.cpp}"

PTO_SKIP_GENERATE="${PTO_SKIP_GENERATE:-1}"

mkdir -p "${ROOT}/build/a3/logs"
stamp="$(date +%Y%m%d_%H%M%S)"
log="${ROOT}/build/a3/logs/${SUBDIR//\//_}_${CPP_GLOB//\*/ALL}_${MODE}_${stamp}.log"

case "${MODE}" in
  sim|npu)
    if [[ -z "${PTO_SOC_VERSION:-}" ]]; then
      if [[ "${MODE}" == "sim" ]]; then
        PTO_SOC_VERSION="Ascend910B1"
      else
        PTO_SOC_VERSION="Ascend910B"
      fi
    fi
    PTO_RUN_MODE="${MODE}" \
    PTO_SOC_VERSION="${PTO_SOC_VERSION}" \
    PTO_SKIP_GENERATE="${PTO_SKIP_GENERATE}" \
    PTO_SUBDIRS="${SUBDIR}" \
    PTO_CPP_GLOB="${CPP_GLOB}" \
    "${ROOT}/scripts/a3/build_and_run_examples.sh" |& tee "${log}"
    ;;
  *)
    echo "Usage: $0 [sim|npu] [subdir] [cpp_glob]"
    echo "Examples:"
    echo "  $0 sim fused_softmax rowmax.cpp"
    echo "  $0 npu torch_nn nn_Softmax.cpp"
    exit 2
    ;;
esac

echo "Log: ${log}"
