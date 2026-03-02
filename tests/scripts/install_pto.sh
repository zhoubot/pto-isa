#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

TOOLKIT_INSTALL_PATH=""
TOOLKIT_PACKAGE_PATH=""
DOWNLOAD_DIR="./downloads"
PTO_ISA_PKG=""
GTEST_PKG=""
LOG_FILE="./setup_pto_isa_env.log"

CURRENT_DATE=$(date '+%Y%m%d')
CURRENT_YEAR=$(date '+%Y')
CURRENT_MONTH=$(date '+%m')

MAX_BACKTRACK_DAYS=3

log() {
  local level="$1"
  local message="$2"
  echo -e "${level}[$(date '+%Y-%m-%d %H:%M:%S')] $message${NC}" | tee -a "$LOG_FILE"
}

info() { log "${GREEN}[INFO]${NC}" "$1"; }
warn() { log "${YELLOW}[WARN]${NC}" "$1"; }
error() { log "${RED}[ERROR]${NC}" "$1"; }

detect_os() {
  uname -s
}

detect_arch() {
  local arch=$(uname -m)
  case $arch in
    aarch64|arm64)
      echo "aarch64"
      ;;
    x86_64|amd64)
      echo "x86_64"
      ;;
    *)
      error "not support arch: $arch"
      exit 1
      ;;
  esac
}

check_toolkit_installed() {
  local path="$1"
  local setenv_path1="${path}/ascend-toolkit/set_env.sh"
  local setenv_path2="${path}/cann/bin/setenv.bash"

  if [[ -f "$setenv_path1" || -f "$setenv_path2" ]]; then
    info "toolkit installed: $path"
    return 0
  else
    warn "toolkit has not installed"
    return 1
  fi
}

install_toolkit() {
  local package_path="$1"
  local install_path="$2"

  if [[ ! -f "$package_path" ]]; then
    error "toolkit package not exist: $package_path"
    exit 1
  fi
  info "install toolkit package to $install_path"

  if [[ "$package_path" == *.run ]]; then
    chmod +x "$package_path"
    "$package_path" --full --quiet --install-path="$install_path"
  else
    error "not support format: $package_path"
    exit 1
  fi

  if [[ $? -ne 0 ]]; then
    error "install toolkit failed!"
    exit 1
  fi

  info "install toolkit success!"
}

download_file() {
  local url="$1"
  local out="$2"

  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --silent --show-error --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url" -q
  else
    error "need curl or wget to download: $url"
    exit 1
  fi
}

cpu_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.ncpu
  else
    echo 1
  fi
}

gen_try_dates() {
  if command -v python3 >/dev/null 2>&1; then
    python3 - <<PY
import datetime
start = datetime.datetime.strptime("${CURRENT_DATE}", "%Y%m%d").date()
for i in range(${MAX_BACKTRACK_DAYS}):
    print((start - datetime.timedelta(days=i)).strftime("%Y%m%d"))
PY
    return 0
  fi

  if date -d "${CURRENT_DATE}" +%s >/dev/null 2>&1; then
    local current
    current=$(date -d "${CURRENT_DATE}" +%s)
    for i in $(seq 0 $((MAX_BACKTRACK_DAYS - 1))); do
      date -d "@$((current - i * 86400))" '+%Y%m%d'
    done
    return 0
  fi

  error "failed to generate backtrack dates (need python3 or GNU date)"
  exit 1
}

download_pto_isa_run() {
  local arch="$1"
  local base_url="http://container-obsfs-filesystem.obs.cn-north-4.myhuaweicloud.com/package/cann/pto-isa/version_compile/master/${CURRENT_YEAR}${CURRENT_MONTH}"

  local filename="cann-pto-isa_8.5.0_linux-${arch}.run"
  PTO_ISA_PKG="${DOWNLOAD_DIR}/${filename}"

  mkdir -p  "$DOWNLOAD_DIR"

  if [[ -f "$PTO_ISA_PKG" ]]; then
    info "pto isa package exist: $PTO_ISA_PKG"
    return 0
  fi

  while IFS= read -r date; do
    local url="${base_url}/${date}/ubuntu_${arch}/${filename}"

    info "downloading $url"

    if download_file "$url" "$PTO_ISA_PKG"; then
      info "download successful!"
      return 0
    else
      warn "download failed!: $url"
    fi
  done < <(gen_try_dates)

  error "please check network!"
  exit 1
}

install_pto_isa_run() {
  local toolkit_path="$1"
  local install_path="${toolkit_path}"

  chmod +x "$PTO_ISA_PKG"
  "$PTO_ISA_PKG" --full --quiet --install-path="$install_path"

  if [[ $? -ne 0 ]]; then
    error "install PTO ISA package failed!"
    exit 1
  fi
  info "install PTO ISA package success!"
}

install_gtest() {
  local src_dir="${DOWNLOAD_DIR}/googletest"
  local build_dir="${DOWNLOAD_DIR}/googletest-build"

  if [[ -d "$src_dir/.git" ]]; then
    info "googletest repo exists: $src_dir"
  else
    rm -rf "$src_dir"
    git clone --depth 1 --branch v1.14.0 https://github.com/google/googletest.git "$src_dir"
  fi

  rm -rf "$build_dir"
  cmake -S "$src_dir" -B "$build_dir" -DCMAKE_CXX_FLAGS="-fPIC"
  cmake --build "$build_dir" --parallel "$(cpu_jobs)"

  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    cmake --install "$build_dir"
    info "install gtest success (as root)"
    return 0
  fi

  if command -v sudo >/dev/null 2>&1; then
    sudo cmake --install "$build_dir"
    info "install gtest success (via sudo)"
    return 0
  fi

  warn "skip installing gtest: need root or sudo"
}

main() {
  local OS
  OS="$(detect_os)"
  if [[ "$OS" != "Linux" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
      TOOLKIT_INSTALL_PATH="${1:-${HOME}/Ascend}"
      if [[ -n "${2:-}" ]]; then
        warn "ignoring toolkit_package_path on macOS (CANN toolkit .run is Linux-only)"
      fi

      if mkdir -p "$TOOLKIT_INSTALL_PATH" 2>/dev/null; then
        info "install prefix (macOS user-mode): $TOOLKIT_INSTALL_PATH"
      else
        warn "cannot write to: $TOOLKIT_INSTALL_PATH"
        TOOLKIT_INSTALL_PATH="${HOME}/Ascend"
        mkdir -p "$TOOLKIT_INSTALL_PATH"
        info "fallback install prefix: $TOOLKIT_INSTALL_PATH"
        warn "to use /usr/local/Ascend on macOS, create it with admin privileges first"
      fi

      warn "Ascend CANN / NPU runtime is Linux-only; macOS supports CPU simulator workflows only"
      info "running CPU simulator tests..."
      python3 tests/run_cpu.py --clean --verbose
      info "set environment successfully (CPU simulator)"
      return 0
    fi

    error "unsupported OS: $OS"
    exit 1
  fi

  if [[ $# -lt 1 ]]; then
    error "usage: $0 <toolkit_install_path> [toolkit_package_path]"
    exit 1
  fi

  TOOLKIT_INSTALL_PATH="$1"
  TOOLKIT_PACKAGE_PATH="${2:-}"

  local ARCH=$(detect_arch)
  info "arch:$ARCH"

  if check_toolkit_installed "$TOOLKIT_INSTALL_PATH"; then
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$TOOLKIT_INSTALL_PATH"
  else
    if [[ -z "$TOOLKIT_PACKAGE_PATH" ]]; then
      error "no toolkit package info"
      exit 1
    fi

    install_toolkit "$TOOLKIT_PACKAGE_PATH" "$TOOLKIT_INSTALL_PATH"
    download_pto_isa_run "$ARCH"
    install_pto_isa_run "$TOOLKIT_INSTALL_PATH"
  fi

  install_gtest

  info "set environment successfully!"
}

main "$@"
