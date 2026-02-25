#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/ptoas_bin.sh [auto|pack|unpack|split|join|status] [--force]

Purpose:
  Manage the vendored LLVM `ptoas` binary in `bin/`:
    - Pack `bin/ptoas` into `bin/ptoas.tar.xz` (or `.tar.gz` fallback)
    - Unpack the tarball back into `bin/ptoas`
    - Split the tarball into chunks (<10MB by default) for repos that disallow large files

Behavior (auto):
  - If `bin/ptoas` exists but no tarball exists: create a tarball.
  - If `bin/ptoas` does NOT exist but a tarball exists: extract the tarball.
  - If both exist: do nothing.

Notes:
  - This script does not download `ptoas`. It only packs/unpacks local files.
  - Prefer `.tar.xz` when available (smaller). Falls back to `.tar.gz`.
  - Split parts are named:
      `bin/ptoas.tar.xz.part000`, `bin/ptoas.tar.xz.part001`, ...
    Reassemble with:
      `bash scripts/ptoas_bin.sh join`
  - Control split size via `PTOAS_TARBALL_CHUNK_BYTES` (default: 9m).
EOF
}

cmd="${1:-auto}"
shift || true

force="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) force="true"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

repo_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1
  pwd
)"
bin_dir="${repo_root}/bin"
ptoas_bin="${bin_dir}/ptoas"
ptoas_tar_xz="${bin_dir}/ptoas.tar.xz"
ptoas_tar_gz="${bin_dir}/ptoas.tar.gz"

have_bin() { [[ -f "${ptoas_bin}" ]]; }
have_xz() { [[ -f "${ptoas_tar_xz}" ]]; }
have_gz() { [[ -f "${ptoas_tar_gz}" ]]; }

have_xz_parts() { compgen -G "${ptoas_tar_xz}.part*" >/dev/null 2>&1; }
have_gz_parts() { compgen -G "${ptoas_tar_gz}.part*" >/dev/null 2>&1; }

join_parts() {
  local archive="$1"
  local tmp="${archive}.tmp"
  local parts=()
  while IFS= read -r p; do parts+=("$p"); done < <(ls -1 "${archive}.part"* 2>/dev/null | sort || true)
  if [[ ${#parts[@]} -eq 0 ]]; then
    echo "error: no split parts found for: ${archive}.part*" >&2
    return 1
  fi
  if [[ -f "${archive}" && "${force}" != "true" ]]; then
    return 0
  fi
  echo "[ptoas_bin] joining ${#parts[@]} parts -> ${archive}"
  cat "${parts[@]}" > "${tmp}"
  mv -f "${tmp}" "${archive}"
}

split_archive() {
  local archive="$1"
  local chunk="${PTOAS_TARBALL_CHUNK_BYTES:-9m}"
  if [[ ! -f "${archive}" ]]; then
    echo "error: missing tarball: ${archive}" >&2
    return 1
  fi
  if ! command -v split >/dev/null 2>&1; then
    echo "error: 'split' not found in PATH" >&2
    return 1
  fi
  if compgen -G "${archive}.part*" >/dev/null 2>&1; then
    if [[ "${force}" != "true" ]]; then
      return 0
    fi
    rm -f "${archive}.part"* || true
  fi
  echo "[ptoas_bin] splitting (${chunk}) -> ${archive}.partNNN"
  split -b "${chunk}" -d -a 3 "${archive}" "${archive}.part"
}

pack_xz() {
  mkdir -p "${bin_dir}"
  if ! have_bin; then
    echo "error: missing binary: ${ptoas_bin}" >&2
    return 1
  fi
  if have_xz && [[ "${force}" != "true" ]]; then
    return 0
  fi
  echo "[ptoas_bin] packing -> ${ptoas_tar_xz}"
  tar -C "${bin_dir}" -cJf "${ptoas_tar_xz}.tmp" "ptoas"
  mv -f "${ptoas_tar_xz}.tmp" "${ptoas_tar_xz}"
}

pack_gz() {
  mkdir -p "${bin_dir}"
  if ! have_bin; then
    echo "error: missing binary: ${ptoas_bin}" >&2
    return 1
  fi
  if have_gz && [[ "${force}" != "true" ]]; then
    return 0
  fi
  echo "[ptoas_bin] packing -> ${ptoas_tar_gz}"
  tar -C "${bin_dir}" -czf "${ptoas_tar_gz}.tmp" "ptoas"
  mv -f "${ptoas_tar_gz}.tmp" "${ptoas_tar_gz}"
}

unpack_xz() {
  mkdir -p "${bin_dir}"
  if ! have_xz; then
    echo "error: missing tarball: ${ptoas_tar_xz}" >&2
    return 1
  fi
  if have_bin && [[ "${force}" != "true" ]]; then
    return 0
  fi
  echo "[ptoas_bin] unpacking <- ${ptoas_tar_xz}"
  tar -C "${bin_dir}" -xJf "${ptoas_tar_xz}"
  chmod 0755 "${ptoas_bin}" || true
}

unpack_gz() {
  mkdir -p "${bin_dir}"
  if ! have_gz; then
    echo "error: missing tarball: ${ptoas_tar_gz}" >&2
    return 1
  fi
  if have_bin && [[ "${force}" != "true" ]]; then
    return 0
  fi
  echo "[ptoas_bin] unpacking <- ${ptoas_tar_gz}"
  tar -C "${bin_dir}" -xzf "${ptoas_tar_gz}"
  chmod 0755 "${ptoas_bin}" || true
}

choose_pack() {
  # Prefer xz when tar supports -J (and xz is installed).
  if tar --help 2>/dev/null | grep -q -- " -J, --xz"; then
    pack_xz
    return 0
  fi
  pack_gz
}

choose_unpack() {
  if have_xz; then
    unpack_xz
    return 0
  fi
  if have_gz; then
    unpack_gz
    return 0
  fi
  if have_xz_parts; then
    join_parts "${ptoas_tar_xz}"
    unpack_xz
    return 0
  fi
  if have_gz_parts; then
    join_parts "${ptoas_tar_gz}"
    unpack_gz
    return 0
  fi
  echo "error: no tarball found (expected ${ptoas_tar_xz} or ${ptoas_tar_gz})" >&2
  return 1
}

status() {
  echo "repo_root: ${repo_root}"
  echo "bin:       ${ptoas_bin} $(have_bin && echo '[OK]' || echo '[MISSING]')"
  echo "tar.xz:    ${ptoas_tar_xz} $(have_xz && echo '[OK]' || echo '[MISSING]')"
  echo "tar.xz.*:  ${ptoas_tar_xz}.part* $(have_xz_parts && echo '[OK]' || echo '[MISSING]')"
  echo "tar.gz:    ${ptoas_tar_gz} $(have_gz && echo '[OK]' || echo '[MISSING]')"
  echo "tar.gz.*:  ${ptoas_tar_gz}.part* $(have_gz_parts && echo '[OK]' || echo '[MISSING]')"
}

case "${cmd}" in
  -h|--help|help)
    usage
    ;;
  status)
    status
    ;;
  pack)
    choose_pack
    ;;
  unpack)
    choose_unpack
    ;;
  split)
    if ! have_xz && ! have_gz; then
      choose_pack
    fi
    if have_xz; then
      split_archive "${ptoas_tar_xz}"
    else
      split_archive "${ptoas_tar_gz}"
    fi
    ;;
  join)
    if have_xz || have_xz_parts; then
      join_parts "${ptoas_tar_xz}"
    elif have_gz || have_gz_parts; then
      join_parts "${ptoas_tar_gz}"
    else
      echo "error: no tarball parts found to join" >&2
      exit 1
    fi
    ;;
  auto)
    if have_bin; then
      if have_xz || have_gz; then
        exit 0
      fi
      choose_pack
      exit 0
    fi
    if have_xz || have_gz; then
      choose_unpack
      exit 0
    fi
    echo "error: neither ${ptoas_bin} nor a tarball exists" >&2
    exit 1
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    usage
    exit 2
    ;;
esac
