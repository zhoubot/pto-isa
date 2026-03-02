# tests/scripts/

Repository helper tests/scripts, mainly for packaging and release workflows.

## Layout

- `package/`: Packaging tests/scripts (Python + templates + configuration)

## Entry Point

- `build.sh --pkg` triggers the packaging flow implemented under `tests/scripts/package/`

## PTOAS Binary (tarball)

Some workflows vendor the LLVM `ptoas` tool as a tarball (`bin/ptoas.tar.xz`) instead of a raw `bin/ptoas`.

- Pack/unpack helper: `tests/scripts/ptoas_bin.sh` (run `tests/scripts/ptoas_bin.sh --help`)
- For repos with a strict file-size limit, use `tests/scripts/ptoas_bin.sh split` to create `bin/ptoas.tar.xz.partNNN` chunks.
