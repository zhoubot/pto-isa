#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-cpu}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "${SCRIPT_DIR}/run.py" --target "${TARGET}" "$@"
