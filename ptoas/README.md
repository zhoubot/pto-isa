# `ptoas`: PTO Assembly + MLIR dialect prototype

This folder contains:

- A draft MLIR dialect definition (TableGen) for PTO Assembly / PTO intrinsics.
- A consolidated IR specification (`ptoas/IR_SPEC.md`) describing `make_tensor_view` / `partition_view` / `tload` semantics.
- Utilities to migrate documentation examples from SSA-style PTO-AS to a destination-passing style (DPS) syntax.
- A prototype end-to-end pipeline that can translate a `*.pto` program into an AscendC kernel that calls PTO intrinsics, and build it with `bisheng` to produce a `*.bin`.

Status:

- The dialect in `ptoas/PTOAS.td` is a **spec prototype** (TableGen only). Integrating it into a real `mlir-opt`/pass pipeline requires MLIR toolchain support in the build environment.
- The pipeline script (`ptoas/tools/ptoas_build.py`) is intentionally minimal and only supports a small subset of PTO-AS sufficient for a demo (`tassign`, `tload`, `tadd`, `tstore`, `mgather`, `mscatter`).

Quick start (A2/A3 vector core):

```bash
python3 ptoas/tools/ptoas_build.py ptoas/examples/add16.pto --arch dav-c220-vec
```

Type spelling notes:

- Use `!pto.tensor<dtype=..., shape=[...], stride=[...], layout=...>` (older `!pto.gtensor<element=...>` is accepted by the Python prototype).
- Use `!pto.tile<dtype=..., ...>` (older `element=...` is accepted by the Python prototype).
