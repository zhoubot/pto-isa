#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.."; pwd)"
MODE="${1:-sim}" # sim|npu

mkdir -p "${ROOT}/build/a3/logs"
log="${ROOT}/build/a3/logs/rowmax_${MODE}.log"

case "${MODE}" in
  sim)
    if [[ -z "${PTO_SOC_VERSION:-}" ]]; then
      PTO_SOC_VERSION="Ascend910B1"
    fi
    PTO_RUN_MODE=sim \
    PTO_SOC_VERSION="${PTO_SOC_VERSION}" \
    PTO_SKIP_GENERATE=1 \
    PTO_SUBDIRS=fused_softmax \
    PTO_CPP_GLOB=rowmax.cpp \
    "${ROOT}/scripts/a3/build_and_run_examples.sh" |& tee "${log}"
    ;;
  npu)
    PTO_RUN_MODE=npu \
    PTO_SKIP_GENERATE=1 \
    PTO_SUBDIRS=fused_softmax \
    PTO_CPP_GLOB=rowmax.cpp \
    "${ROOT}/scripts/a3/build_and_run_examples.sh" |& tee "${log}"
    ;;
  *)
    echo "Usage: $0 [sim|npu]"
    exit 2
    ;;
esac

echo "Log: ${log}"
