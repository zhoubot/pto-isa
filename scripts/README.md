# scripts/

Repository helper scripts, mainly for packaging and release workflows.

## Layout

- `package/`: Packaging scripts (Python + templates + configuration)

## Entry Point

- `build.sh --pkg` triggers the packaging flow implemented under `scripts/package/`

## PTOAS Binary (tarball)

Some workflows vendor the LLVM `ptoas` tool as a tarball (`bin/ptoas.tar.xz`) instead of a raw `bin/ptoas`.

- Pack/unpack helper: `scripts/ptoas_bin.sh` (run `scripts/ptoas_bin.sh --help`)
- For repos with a strict file-size limit, use `scripts/ptoas_bin.sh split` to create `bin/ptoas.tar.xz.partNNN` chunks.
