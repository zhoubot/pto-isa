#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/install_cann_toolkit.sh --installer <path/to/Ascend-cann-toolkit_*.run> [options]

Options:
  --install-path <path>   Install prefix (default: /usr/local/Ascend if root, else $HOME/Ascend)
  --mode <install|devel|full>   Install mode (default: full)
  --chip <chip_type>      e.g. Ascend910B, Ascend910_93, Ascend310B, Ascend310P
  --whitelist <features>  e.g. nnae,nnrt,hccl,atc,devtools,hccl-only
  --feature <features>    e.g. ascendc
  --force                 Force install/upgrade
  --install-for-all       Install for all users (typically requires root)
  --quiet                 Quiet mode (accepts EULA, skips prompts)
  --check                 Run installer integrity/version checks first
  -h, --help              Show help

Notes:
  - This installer is Linux-only (aarch64/x86_64). On macOS this script exits with guidance.
  - For extraction-only (no install), run:
      bash <installer.run> --noexec --extract <dir>
EOF
}

installer=""
install_path=""
mode="full"
chip=""
whitelist=""
feature=""
force="false"
install_for_all="false"
quiet="false"
do_check="false"

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --installer)
      installer="${2:-}"; shift 2 ;;
    --install-path)
      install_path="${2:-}"; shift 2 ;;
    --mode)
      mode="${2:-}"; shift 2 ;;
    --chip)
      chip="${2:-}"; shift 2 ;;
    --whitelist)
      whitelist="${2:-}"; shift 2 ;;
    --feature)
      feature="${2:-}"; shift 2 ;;
    --force)
      force="true"; shift ;;
    --install-for-all)
      install_for_all="true"; shift ;;
    --quiet)
      quiet="true"; shift ;;
    --check)
      do_check="true"; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$installer" ]]; then
  echo "Missing required --installer" >&2
  usage
  exit 2
fi

if [[ ! -f "$installer" ]]; then
  echo "Installer not found: $installer" >&2
  exit 1
fi

os="$(uname -s)"
if [[ "$os" != "Linux" ]]; then
  cat >&2 <<EOF
This installer is Linux-only.

You are on: $os

From macOS you can:
  - copy the .run to an ARM64 Linux machine/VM (or Ascend host) and run this script there, or
  - extract it locally for inspection:
      bash "$installer" --noexec --extract ./toolkit_tmp
EOF
  exit 1
fi

if [[ -z "$install_path" ]]; then
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    install_path="/usr/local/Ascend"
  else
    install_path="${HOME}/Ascend"
  fi
fi

case "$mode" in
  install|devel|full) ;;
  *)
    echo "Invalid --mode: $mode (expected: install|devel|full)" >&2
    exit 2
    ;;
esac

chmod +x "$installer"

if [[ "$do_check" == "true" ]]; then
  echo "+ $installer --check"
  "$installer" --check
fi

cmd=("$installer" "--${mode}" "--install-path=${install_path}")
if [[ "$quiet" == "true" ]]; then
  cmd+=("--quiet")
fi
if [[ "$force" == "true" ]]; then
  cmd+=("--force")
fi
if [[ "$install_for_all" == "true" ]]; then
  cmd+=("--install-for-all")
fi
if [[ -n "$chip" ]]; then
  cmd+=("--chip=${chip}")
fi
if [[ -n "$whitelist" ]]; then
  cmd+=("--whitelist=${whitelist}")
fi
if [[ -n "$feature" ]]; then
  cmd+=("--feature=${feature}")
fi

printf '+ %q ' "${cmd[@]}"
printf '\n'
exec "${cmd[@]}"

