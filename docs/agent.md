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

#### Event Insertion + `set_flag / wait_flag` (A2/A3 contract)

On A2/A3, inter-pipeline sync is lowered to `set_flag(srcPipe, dstPipe, eventId)` / `wait_flag(srcPipe, dstPipe, eventId)`.
The important compiler-facing rules (simplified) are:

- `set_flag(src,dst,id)` publishes completion of **all prior `src`-pipe ops** (program-order scope) to `dst`.
- `wait_flag(src,dst,id)` blocks `dst` until the matching token is available, then acts like an acquire for the published `src` effects.
- Token balance must hold per `(src,dst,id)`: completed waits must never exceed completed sets.
- Buffer reuse hazards require an explicit reverse dependency `(dst→src)`; otherwise you can get WAR/WAW bugs.
- The 8 `eventId` slots (0..7) are a bounded resource; avoid reusing an id before the previous token is consumed.

Recent `InsertEventsPass` behavior to keep `--insert-events` usable in loops:

- Avoids emitting multiple `wait_flag` against the same producer token (prevents deadlocks on token-consuming hardware).
- Adds loop-carried reverse-dependency “handshakes” around matmul pipelines to protect reuse:
  - `TMATMUL (PIPE_M) -> TMOV_M2L (PIPE_MTE1)` (L0A/L0B reuse)
  - `TMOV_M2L (PIPE_MTE1) -> TLOAD (PIPE_MTE2)` (L1/global reuse)
  - See `ptoas/mlir/lib/InsertEventsPass.cpp`.

#### Recent emitter note: indexed `tload/tstore` tile views

If PTO-AS uses indexed tile accesses like `pto.tload %x[%r, %c]`, the MLIR CCE/CPU emitters must materialize a
tile-shaped `GlobalTensor` view (rows/cols) to keep A2/A3 layout conversions (e.g., ND2NZ / DN2ZN) correct.
This is handled in `ptoas/mlir/lib/CCEmitter.cpp`.

#### PTO ISA coverage note (prototype)

`ptoas` CCE/CPU emission covers a core subset explicitly (`tload/tstore/tmov/tadd/tmatmul/...`).
For most remaining `pto.t*` ops, the emitter falls back to emitting the corresponding uppercase PTO macro
(e.g. `trowmax -> TROWMAX`, `tmuls -> TMULS`) so Python-authored kernels can still compile end-to-end.

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

### Run On Real NPU (CPU Reference + `acl`)

The extracted `*.bin` is a useful build artifact, but `acl.rt.binary_load_from_file(..., [])` may fail with error `107000`
on some setups. A reliable way to validate kernels on real hardware is to build a **fatobj shared library** with `bisheng`
and launch the kernel via `<<<>>>` (same pattern as `tests/npu/*/src/st`).

End-to-end script (compiles the same `*.pto` to **CPU** and **NPU**, runs both, and compares outputs):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_e2e_npu.py --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Inputs used by the script:

- `ptoas/examples/add16_e2e.pto` (Vec add)
- `ptoas/examples/gemm16_e2e.pto` (Cube GEMM: f16xf16->f32)

You can also run the same script on the Ascend simulator:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_e2e_npu.py --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

### Run On CPU Simulator (Emit C++ + `ctypes`)

CPU target emits a shared-library-friendly kernel entry:

- `extern "C" void pto_kernel_cpu(void* arg0, void* arg1, void* arg2)`

End-to-end script (emits `*.cpp`, builds `*.so` with a host C++ compiler, runs on CPU):

```bash
python3 ptoas/tools/run_e2e_cpu.py
```

Notes:

- CPU GEMM uses a CPU-specific tile layout that matches `include/pto/cpu/TMatmul.hpp`:
  - Example: `ptoas/examples/gemm16_cpu.pto`

### Python Frontend (“Binding”) For PTO-AS (Cross-Platform)

There is also a small Python “binding” layer that can **generate PTO-AS** and drive the full toolchain:

- Low-level PTO-AS builder (generates `*.pto` text): `ptoas/python/pto_asm.py`
- AST-based frontend (Python -> PTO-AS, supports `for`/`if` + tensor arg metadata): `ptoas/python/ast_frontend.py`
- Simple Python binding (compile `foo.py` -> `foo.pto`): `ptoas/python/binding.py`
- DSL stubs for IDE/type checking (not executable): `ptoas/python/dsl.py`
- Shared compile/run helpers: `ptoas/python/pipeline.py`
- Host C++ launcher generator (emits `host.cpp` that calls `ptoas_launch` from a fatobj `.so`): `ptoas/python/host_codegen.py`
- End-to-end runner (Python frontend -> `*.pto` -> `ptoas` -> CPU+NPU run -> compare):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_python_frontend_e2e.py --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Simulator variant:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 ptoas/tools/run_python_frontend_e2e.py --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Minimal `foo.py` -> `foo.pto` (includes a small host metadata block at the top of the file):

```bash
python3 ptoas/tools/python_to_pto.py ptoas/examples/python_kernels.py --kernel add16 --out /tmp/add16.pto
```

Recommended kernel authoring style (object DSL):

- Import: `from pto_as import PTO, scalar`
- Declare args/tiles with explicit names:
  - `q = pto.tensor("q", (S, D), dtype="f32")`
  - `q_tile = pto.vec_tile("q_tile", dtype="f32", shape=(S, D))`
- Use dest-first instruction calls:
  - `pto.tload(q_tile, q)` (defaults to `[0,0]`)
  - `pto.tstore(o, out_acc)` (defaults to `[0,0]`)
- Compile-time constants via `pto.const(...)` support basic arithmetic and `sqrt(...)` folding (AST-only).
- Python variables can alias PTO-AS names, e.g. `centered = pto.vec_tile("scores_centered", ...)`.

Example: `ptoas/examples/pypto_flash_attention.py`

The metadata block looks like:

- `; PTO_HOST_SPEC_BEGIN v1`
- JSON payload (arg dtypes/shapes/roles, seed, blockDim)
- `; PTO_HOST_SPEC_END`

There is also a file-driven flow that takes a Python kernel file and emits:

- `foo.pto`
- `foo.cpu.cpp` (CPU kernel source)
- `foo.npu.cpp` + `foo.npu.bin` (NPU kernel source + extracted binary)
- `host.cpp` (optional ACL launcher template)

Example (uses `ptoas/examples/python_kernels.py`):

```bash
python3 ptoas/tools/python_kernel_flow.py ptoas/examples/python_kernels.py --kernel add16 --outdir /tmp/ptoas_py_kernel
python3 ptoas/tools/python_kernel_flow.py ptoas/examples/python_kernels.py --kernel gemm16 --arch dav-c220-cube --build-so --ascend-home "$ASCEND_HOME_PATH"
```

### Run On NPU Simulator (`sim` / A3) (Python Kernel → Compare Against CPU)

This path runs the kernel via Ascend simulator libraries (same SoC mapping as `tests/script/run_st.py`).

If you see `LLVM ERROR: make_tensor_view expects 1 operand (dest)`, your `ptoas` binary is out of date.
Rebuild it with:

```bash
bash ptoas/mlir/scripts/build_all.sh
```

Prereqs:

- Install and source Ascend toolkit env (sets `ASCEND_HOME_PATH`):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
source "$ASCEND_HOME_PATH/bin/setenv.bash"
```

Run (Vec add, A3 simulator):

```bash
python3 ptoas/tools/run_python_kernel_sim_e2e.py \
  --py ptoas/examples/python_kernels.py --kernel add16 \
  --soc a3 --arch dav-c220-vec \
  --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Run (Cube GEMM, A3 simulator):

```bash
python3 ptoas/tools/run_python_kernel_sim_e2e.py \
  --py ptoas/examples/python_kernels.py --kernel gemm16 \
  --soc a3 --arch dav-c220-cube \
  --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Outputs go to `/tmp/ptoas_py_kernel_sim` by default and include:

- `*.pto` (PTO-AS)
- `*.cpu.cpp` (CPU source)
- `*.cpp` (NPU CCE source)
- `*.bin` (extracted `__aicore_rel_binary`)
- `lib*_sim.so` (fatobj .so used for launch)
- `host.cpp` (standalone C++ launcher template)

### Kernels: `kernels/python/gemm_big` (sim validated)

`kernels/python/gemm_big` is a larger Python-authored GEMM that relies on `--insert-events` (no manual event ops).

Important layout note:

- The kernel treats `B` as a `DN` view. Host code must pass `b_dev = np.ascontiguousarray(b.T)` (see `kernels/python/gemm_big/run.py`).

Simulator run (example):

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 kernels/python/gemm_big/run.py \
  --run-mode sim --soc a3 \
  --m 256 --n 256 --k 256 \
  --bm 128 --bn 128 --bk 64 \
  --block-dim 4
```

### Python ST Runner (CPU + NPU)

Simple Python ST runner for the `ptoas` toolchain:

```bash
python3 ptoas/tools/run_py_st.py --list
python3 ptoas/tools/run_py_st.py --case add16 --target both --ascend-home "$ASCEND_HOME_PATH" --device 0
python3 ptoas/tools/run_py_st.py --case gemm16 --target both --ascend-home "$ASCEND_HOME_PATH" --device 0
```

Simulator variants:

```bash
python3 ptoas/tools/run_py_st.py --case add16 --target npu --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0
python3 ptoas/tools/run_py_st.py --case gemm16 --target npu --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0
```

### Kernels: Python GEMM Example (CPU + NPU)

There is also a kernel-style example under `kernels/` that uses Python to generate PTO-AS and runs the full toolchain:

```bash
python3 kernels/custom/gemm_python/run.py --target cpu
python3 kernels/custom/gemm_python/run.py --target npu --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
python3 kernels/custom/gemm_python/run.py --target npu --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
python3 kernels/custom/gemm_python/run.py --target both --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

There are also Python-first examples that always emit:

- `*.pto` (PTO-AS)
- `foo.cpp` (CCE source, compiled by `bisheng -xcce` on real NPU env)
- `host.cpp` (a standalone C++ launcher for the fatobj `.so`)

```bash
python3 kernels/python/fa/run.py --target cpu
python3 kernels/python/gemm/run.py --target cpu

# Simulator + compare against CPU
python3 kernels/python/fa/run.py --target npu --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
python3 kernels/python/gemm/run.py --target npu --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

### Kernels: Python Examples Regression (CPU + NPU)

End-to-end examples written as Python kernels under `kernels/python/*.py` (flat layout, compiled by the AST frontend)
can be run as a small regression suite. This folder also carries simplified, runnable ports of the upstream
`~/github/pto-isa/examples/*.py` filenames (e.g. `kernels/python/pto_isa_sinh.py`, `kernels/python/pto_fused_softmax.py`).

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest

# Simulator (A3)
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH"

# Real NPU
python3 kernels/python/run_regression.py --run-mode npu --ascend-home "$ASCEND_HOME_PATH"

# Run a subset
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --cases add16,softmax16,gemm16

# Show compiler output (otherwise build logs are hidden for tidy progress)
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --ascend-home "$ASCEND_HOME_PATH" --verbose-build
```

Notes:

- `ptoas/python/ast_frontend.py` ignores Python docstrings / bare string expression statements inside the kernel body.
  Use `# ...` comments or `pto.comment("...")` for remarks you want to preserve in the generated PTO-AS.

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
