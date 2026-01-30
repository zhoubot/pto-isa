# Tutorial: Python Frontend → PTO → ptoas → bisheng → NPU (with timeout + simulator fallback)

This repo supports writing kernels in a small Python DSL, compiling them into PTO-AS (`.pto`), lowering to Ascend CCE (`.cpp`), building a fatobj (`.so`), and running on NPU via ACL (or on the simulator).

## Prereqs

- Run commands from the repo root.
- `python3` and `numpy`.
- Built `ptoas` at `bin/ptoas` (from `~/llvm-project/build-mlir/bin/ptoas`).
- Ascend toolkit installed and `ASCEND_HOME_PATH` set (and Python can `import acl`).

Recommended env vars:

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
export PTOAS=$PWD/bin/ptoas
export PYTHONPATH="$PWD/binding/python:${PYTHONPATH:-}"
```

## Stage 0: Write the kernel (Python)

Kernels are **parsed (not executed)** by the AST frontend (`binding/python/ptoas/python/ast_frontend.py`), so keep to the supported subset:

- straight-line statements
- `for ... in range(...)`
- simple `if a < b` / `==` compares

Recommended style is the object DSL (`pto_as.PTO`) with “grammar candy”:

- omit repeated explicit names:
  - `x = pto.tensor(dtype="f16", shape=(16, 16), role="in")`
  - `tx = pto.vec(dtype="f16", shape=(16, 16))` (shorthand for `vec_tile`)
- use readable aliases:
  - `tx = pto.load(x)` / `pto.store(y, tx)`
  - `a_left = pto.mov(a_mat)`

Examples:

- `kernels/python/add16.py`
- `kernels/python/gemm16.py`

## Stage 1: Python → PTO-AS (`.pto`)

Generate a `.pto` (with an embedded host-spec header for runners):

```bash
python3 ptoas/tools/python_to_pto.py kernels/python/add16.py --kernel add16 --out /tmp/add16.pto
```

The header is a comment block:

- `// PTO_HOST_SPEC_BEGIN v1`
- JSON payload (arg shapes/dtypes/roles + seed + blockDim)
- `// PTO_HOST_SPEC_END`

## Stage 2: PTO-AS → CCE (`.cpp`)

Compile to NPU CCE source:

```bash
$PTOAS /tmp/add16.pto -o /tmp/add16.cpp
```

Notes:

- The typical run path uses a fatobj `.so` (Stage 3) built via `bisheng`, instead of directly loading `.bin`.
- If you want the compiler to insert synchronization heuristically, add `--enable-insert-sync`.

## Stage 2.1: Multi-stage kernels (cube + vec)

Some kernels want to mix **cube** and **vec** code paths (e.g. `matmul` + `softmax`) in one logical Python program.
In that case, compile with:

- stage markers in the Python program

How it works (high level):

- Python emits marker ops like `pto.stage_qk_cube()` / `pto.stage_softmax_vec()`.
- The Python frontend splits the program into multiple MLIR functions, and `ptoas` emits multiple CCE kernels:
  - `<name>_stage_<stage0>`
  - `pto_kernel_<name>_<stage1>`
  - ...
- `build_fatobj_so_from_cce(...)` builds one stage fatobj `.so` per kernel and a small host wrapper `.so` that
  dlopens and chains them in order.

Example:

```bash
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --cases flash_attention64_split --timeout-sec 600
python3 kernels/python/run_regression.py --run-mode npu --cases flash_attention64_split --timeout-sec 300
```

Set `PTOAS_SPLIT_TRACE=1` to print the stage `.so` names at runtime.

## Stage 3: CCE → fatobj `.so` (via `bisheng -xcce`)

Use the helper (wraps `bisheng -xcce` with the right include/lib flags):

```bash
python3 - <<'PY'
from pathlib import Path
from ptoas.python import pipeline

ascend = pipeline.default_ascend_home()
pipeline.build_fatobj_so_from_cce(
    cce_path=Path("/tmp/add16.cpp"),
    out_so=Path("/tmp/libadd16_npu.so"),
    arch="dav-c220-vec",
    ascend_home=ascend,
)
print("built:", "/tmp/libadd16_npu.so")
PY
```

## Stage 4: Run on NPU (or simulator) and verify

Recommended: use the end-to-end runner (CPU reference + NPU run + compare), with a wall-time timeout:

```bash
python3 ptoas/tools/python_kernel_e2e.py kernels/python/add16.py --kernel add16 --arch dav-c220-vec --run-mode npu --timeout-sec 120
python3 ptoas/tools/python_kernel_e2e.py kernels/python/gemm16.py --kernel gemm16 --arch dav-c220-cube --run-mode npu --timeout-sec 120
```

If a run hangs (deadlock), rerun automatically under the simulator:

```bash
python3 ptoas/tools/python_kernel_e2e.py kernels/python/gemm16.py --kernel gemm16 --arch dav-c220-cube \
  --run-mode npu --timeout-sec 120 --sim-on-timeout --soc a3
```

Artifacts in `--outdir` include:

- `*.pto` (PTO-AS)
- `*.cpu.cpp` + `lib*_cpu.so` (CPU reference)
- `*.cpp` + `*.bin` + `lib*_{npu,sim}.so` (NPU build products)
- `event_summary.txt` and `set_wait_snippet.txt` (best-effort set/wait flag summaries)

## Batch regression (NPU)

Run the curated `kernels/python/*.py` suite:

```bash
python3 kernels/python/run_regression.py --run-mode npu --ascend-home "$ASCEND_HOME_PATH"
```

Add a per-case timeout (kills hung runs by spawning a subprocess per case):

```bash
python3 kernels/python/run_regression.py --run-mode npu --timeout-sec 180 --ascend-home "$ASCEND_HOME_PATH"
```

If your NPU runtime is occasionally flaky, add retries:

```bash
python3 kernels/python/run_regression.py --run-mode npu --timeout-sec 180 --retries 2 --ascend-home "$ASCEND_HOME_PATH"
```

### Note: multi-tile kernels and tile reuse

`ptoas --enable-insert-sync` is a conservative prototype. If a kernel processes multiple tiles back-to-back, avoid reusing
the *same* GM→tile load buffers (`tload` dst) and tile→GM store buffers (`tstore` src) across tiles.

For **looped** multi-tile kernels, you may hit WAR/WAW hazards when reusing the same on-chip tiles across iterations
(e.g. a later `tload` overwriting a buffer while an earlier `tstore` is still reading it). Prefer relying on
`ptoas --enable-insert-sync` to insert the required synchronization.

For example, prefer:

- `tx0/tx1` (ping-pong GM→Vec tiles) instead of reusing one `tx`
- `out0/out1` (ping-pong Vec→GM tiles) instead of reusing one `out`

See `kernels/python/softmax32x16.py` and `kernels/python/pto_llama7B_dynamic.py` for a minimal pattern.

Worked looped examples (SIM + NPU):

- `kernels/python/tiled_add128.py`
- `kernels/python/tiled_transpose64.py`
- `kernels/python/tiled_rowsum64.py`
