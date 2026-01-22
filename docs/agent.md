# Repo Context for AI Agents (PTO Tile Lib)

This document is a fast, practical orientation for agents working in this repo: what it is, where the key entrypoints live, and the shortest paths to build/run in **CPU**, **NPU simulator (`sim`)**, and **on-board NPU (`npu`)** modes.

## What This Repo Is

- **PTO Tile Library**: C++ headers + implementations for the PTO (Parallel Tile Operation) virtual ISA defined by Ascend CANN.
- Supports multiple backends:
  - **CPU simulation** (cross-platform, no Ascend driver/CANN required).
  - **Ascend NPU** backends split by SoC generation:
    - **A2/A3 family**: `include/pto/npu/a2a3/` (selected via `-v a3` in test scripts).
    - **A5**: `include/pto/npu/a5/`.
- Primary include for upper-layer code: `#include <pto/pto-inst.hpp>` (unified entry header).

## Repo Map (Where To Look First)

- Project overview + common commands: `README.md`
- Detailed setup (CPU first, then NPU): `docs/getting-started.md`
- ISA docs and navigation:
  - `docs/README.md` (ISA guide entry)
  - `docs/isa/` (per-instruction reference)
- Public API headers and backend status table: `include/README.md`
- Core public headers / backend split: `include/pto/README.md`
- Build/package entrypoint: `build.sh`, top-level `CMakeLists.txt`, `cmake/`
- Tests entrypoints:
  - CPU simulator tests: `tests/run_cpu.py`, `tests/run_cpu_tests.sh`
  - NPU ST build/run: `tests/script/run_st.py`, `tests/run_st.sh`
  - Test layout overview: `tests/README.md`
- Demos: `demos/` (CPU demos used by `tests/run_cpu.py --demo ...`)
- Kernels: `kernels/` (self-contained kernel/operator mini-projects)
  - Python GEMM end-to-end example (CPU + NPU): `kernels/custom/gemm_python/`

## Run: CPU Simulator (Recommended First)

CPU simulation is meant to be the “works everywhere” correctness path.

From repo root:

```bash
python3 tests/run_cpu.py --clean --verbose
```

Useful variants:

```bash
python3 tests/run_cpu.py --testcase tadd
python3 tests/run_cpu.py --testcase tadd --gtest_filter 'TADDTest.*'
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

Notes:

- CPU ST uses CMake and GoogleTest; it may download GTest if not installed system-wide.
- Compiler requirement is **C++23** (see `tests/cpu/st/CMakeLists.txt`).

## Run: NPU ST (Ascend) — `sim` and `npu`

NPU ST is built/run via `tests/script/run_st.py`:

```bash
python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t <testcase> -g <gtest_filter>
```

Key points:

- `-v a3` selects the **A2/A3** implementation under `include/pto/npu/a2a3/` (the test script maps it to a SoC string like `Ascend910B1`).
- `-r sim` uses the Ascend simulator libraries under `$ASCEND_HOME_PATH/tools/simulator/<SOC>/lib` and `runtime/lib64/stub`.
- `-r npu` runs on real hardware.

Examples (single case):

```bash
python3 tests/script/run_st.py -r sim -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
python3 tests/script/run_st.py -r npu -v a3 -t tadd -g TADDTest.case_float_64x64_64x64
```

Recommended suites (wrapper script):

```bash
chmod +x ./tests/run_st.sh
./tests/run_st.sh a3 sim simple
./tests/run_st.sh a3 npu simple
```

## Environment: Ascend CANN / Toolkit

NPU ST requires a working Ascend environment. Typical setup (choose the correct install path):

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
# or
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
```

`tests/script/run_st.py` expects `ASCEND_HOME_PATH` to be set (usually done by `setenv.bash`).

## Common Pitfalls (And How This Repo Handles Them)

- **GTest ABI mismatch on Linux**: some systems have `libgtest*.a` built with `_GLIBCXX_USE_CXX11_ABI=0`.
  - CPU and NPU ST CMake projects support `PTO_GLIBCXX_USE_CXX11_ABI=auto|0|1` and auto-detect when possible.
- **`sim` open-files limit**: simulator runs may require a higher `ulimit -n` (see `docs/getting-started.md` and `build.sh`).

## PTO-AS + `ptoas` Tooling (Assembly → CCE → BIN)

This repo also contains a prototype PTO assembler toolchain under `ptoas/`.

### PTO-AS Syntax Updates (DPS + SSA-Style Sugar)

PTO-AS is primarily a destination-passing style (DPS) format, but the assembler frontends accept a few MLIR-like
SSA-style conveniences.

Legacy SSA-style result binding (older docs/examples):

```text
%dst = tadd %src0, %src1 : (...) -> ...
```

Canonical DPS form (still accepted):

```text
tadd %dst, %src0, %src1 : (...)
```

New SSA-style *destination sugar* (recommended for readability; still DPS under the hood):

```text
%dst = pto.tadd %src0, %src1 : (...)
%t0  = pto.tload %x[%r0, %c0]
pto.tstore %y[%r0, %c0], %t0
```

Declaration updates:

- Tensors can be introduced via `pto.make_tensor_view` from implicit `%argN` kernel args (instead of `.arg %x : !pto.tensor<...>`).
- Tiles can be introduced via `pto.alloc_tile` (optionally binding an address), replacing the `.arg tile + tassign` pattern.

Key type spelling changes:

- Global memory type renamed from `!pto.gtensor<...>` to `!pto.tensor<...>`.
- The element field is renamed from `element=...` to `dtype=...`.
- Canonical spellings live in `docs/grammar/PTO-AS.md`.

Relevant files:

- Spec/grammar: `docs/grammar/PTO-AS.md`, `docs/grammar/PTO-AS.bnf`
- ISA pages: `docs/isa/*.md` (examples updated to DPS + `!pto.tensor` + `dtype`)
- TableGen prototype: `ptoas/PTOAS.td` (spec-only dialect surface)
  - Auto-generated op stubs: `ptoas/PTOASOps.td` (regen: `python3 ptoas/tools/gen_ptoas_ops_td.py`)

### Python Prototype (`*.pto` → `*.bin`) (No MLIR required)

For quick experiments without MLIR, there is a Python pipeline:

- Script: `ptoas/tools/ptoas_build.py`
- Example input: `ptoas/examples/add16.pto`
- Output: emits `*_kernel.cpp`, compiles with `bisheng`, and extracts `__aicore_rel_binary` into `*.bin`.

Run:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/ptoas_build.py ptoas/examples/add16.pto --arch dav-c220-vec
```

Notes:

- The Python tool accepts the newer `!pto.tensor<dtype=...>` and also tolerates older `!pto.gtensor<element=...>` spellings.

### MLIR-based `ptoas` (Buildable/Distributable Binary)

There is also an MLIR-linked `ptoas` prototype under `ptoas/mlir/`:

- Source + build: `ptoas/mlir/CMakeLists.txt`, `ptoas/mlir/tools/ptoas_main.cpp`
- Frontend: parses PTO-AS into an MLIR module with **unregistered** `pto.*` ops:
  - `ptoas/mlir/lib/PTOASFrontend.cpp`
- Pass prototype: inserts `pto.record_event` + `pto.tsync` between memory/vector op boundaries (heuristic):
  - `ptoas/mlir/lib/InsertEventsPass.cpp`
- Emitter: emits Ascend CCE source from the module:
  - `ptoas/mlir/lib/CCEmitter.cpp`
- Bisheng driver: compiles CCE source and extracts `__aicore_rel_binary` into `.bin`:
  - `ptoas/mlir/lib/BishengDriver.cpp`

`ptoas` supports two emission targets:

- `--target npu`: emit Ascend CCE (source `*.cpp`) and optionally `--emit-bin=...` via `bisheng`.
- `--target cpu`: emit CPU-simulator C++ (`*.cpp`); disables `--insert-events` automatically.

#### Build MLIR + `ptoas`

`mlir-opt` is not assumed to exist system-wide; build it from `~/llvm-project`:

```bash
bash ptoas/mlir/scripts/build_all.sh
```

Outputs:

- `~/llvm-project/build-mlir/bin/mlir-opt`
- `ptoas/mlir/build/bin/ptoas`

#### End-to-end Test (Emit `.cpp` + `.bin`)

Use the minimal test program:

- `ptoas/examples/add16_min.pto`

Run:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
export PTO_REPO_ROOT=$(pwd)

./ptoas/mlir/build/bin/ptoas ptoas/examples/add16_min.pto \
  -o /tmp/add16_min.cpp \
  --insert-events \
  --emit-bin=/tmp/add16_min.bin \
  --arch dav-c220-vec \
  --memory-model MEMORY_BASE \
  --repo-root "$PTO_REPO_ROOT" \
  --ascend-home "$ASCEND_HOME_PATH"
```

This produces:

- `/tmp/add16_min.cpp` (generated kernel source)
- `/tmp/add16_min.bin` (extracted `__aicore_rel_binary`)

### Run On Real NPU (Python + `acl` + `numpy`)

The extracted `*.bin` is a useful build artifact, but `acl.rt.binary_load_from_file(..., [])` may fail with error `107000`
on some setups. A reliable way to validate kernels on real hardware is to build a **fatobj shared library** with `bisheng`
and launch the kernel via `<<<>>>` (same pattern as `tests/npu/*/src/st`).

End-to-end script (builds `*.cpp` + `*.bin`, then builds a `*.so`, launches on NPU, and checks with numpy):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_e2e_npu.py --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Inputs used by the script:

- `ptoas/examples/add16_e2e.pto` (Vec add)
- `ptoas/examples/gemm16_e2e.pto` (Cube GEMM: f16xf16->f32)

### Run On CPU Simulator (Emit C++ + `ctypes` + `numpy`)

CPU target emits a shared-library-friendly kernel entry:

- `extern "C" void pto_kernel_cpu(void* arg0, void* arg1, void* arg2)`

End-to-end script (emits `*.cpp`, builds `*.so` with `clang++`, runs on CPU, checks with numpy):

```bash
python3 ptoas/tools/run_e2e_cpu.py
```

Notes:

- CPU GEMM uses a CPU-specific tile layout that matches `include/pto/cpu/TMatmul.hpp`:
  - Example: `ptoas/examples/gemm16_cpu.pto`

### Python Frontend (“Binding”) For PTO-AS

There is also a small Python “binding” layer that can **generate PTO-AS** and drive the full toolchain:

- Low-level PTO-AS builder (generates `*.pto` text): `ptoas/python/pto_asm.py`
- AST-based frontend (Python -> PTO-AS, supports `for`/`if`): `ptoas/python/ast_frontend.py`
- Shared compile/run helpers: `ptoas/python/pipeline.py`
- Host C++ launcher generator (emits `host.cpp` that calls `ptoas_launch` from a fatobj `.so`): `ptoas/python/host_codegen.py`
- End-to-end runner (Python frontend -> `*.pto` -> `ptoas` -> `*.cpp`/`*.bin` -> NPU run -> numpy check):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_python_frontend_e2e.py --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

### Python ST Runner (CPU + NPU)

Simple Python ST runner for the `ptoas` toolchain:

```bash
python3 ptoas/tools/run_py_st.py --list
python3 ptoas/tools/run_py_st.py --case add16 --target both --ascend-home "$ASCEND_HOME_PATH" --device 0
python3 ptoas/tools/run_py_st.py --case gemm16 --target both --ascend-home "$ASCEND_HOME_PATH" --device 0
```

### Kernels: Python GEMM Example (CPU + NPU)

There is also a kernel-style example under `kernels/` that uses Python to generate PTO-AS and runs the full toolchain:

```bash
python3 kernels/custom/gemm_python/run.py --target cpu
python3 kernels/custom/gemm_python/run.py --target npu --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
python3 kernels/custom/gemm_python/run.py --target both --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

There are also Python-first examples that always emit:

- `*.pto` (PTO-AS)
- `foo.cpp` (CCE source, compiled by `bisheng -xcce` on real NPU env)
- `host.cpp` (a standalone C++ launcher for the fatobj `.so`)

```bash
python3 kernels/python/fa/run.py --target cpu
python3 kernels/python/gemm/run.py --target cpu
```

### Control Flow In PTO-AS (Prototype)

PTO-AS frontend also supports a small subset of MLIR-like SCF control flow (textual blocks):

```text
scf.for %i = 0 to 2 step 1 {
  %cond = pto.icmp_lt %i, 1 : i1
  scf.if %cond {
    ...
  } else {
    ...
  }
}
```
