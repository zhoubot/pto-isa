# pyPTO (Python PTO frontend)

This repo provides a **Python-first** way to author PTO kernels, then run the full toolchain:

`Python kernel` → `PTO-AS (.pto)` → `ptoas` → `CCE C++` → `bisheng` → `fatobj .so` → `ACL launch (NPU/sim)`

The Python source is typically **parsed (AST)** by `binding/python/ptoas/python/ast_frontend.py` and is **not executed**.

## 1. Two authoring styles

### 1.1 Object DSL (recommended, used by `kernels/python/*`)

Import `PTO` and write readable kernels with “grammar candy”:

```python
from pto_as import PTO

def rowmax16():
    pto = PTO("rowmax16")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(16, 16), role="in")
    y = pto.tensor(dtype="f32", shape=(16, 1), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_max = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")

    tx = pto.load(x)
    row_max = pto.rowmax(tx, tmp)    # sugar for `trowmax`
    pto.store(y, row_max)            # sugar for `tstore`

    pto.epilogue()
    return pto.program()
```

The important convention is:

- **Declarations** are normal Python assignments (`x = pto.tensor(...)`, `tx = pto.vec(...)`).
- **Instructions** are written in a readable style:
  - assignment-form: `dst = pto.op(src0, src1, ...)`
  - statement-form: `pto.store(tensor, tile)` / `pto.comment("...")`

### 1.2 Function DSL stubs (`ptoas.python.dsl`)

`binding/python/ptoas/python/dsl.py` contains non-executable stubs for authors that prefer:

```python
from ptoas.python.dsl import *
```

This style is also AST-parsed. The stubs exist mainly for import-time names and IDE completion.

## 2. Naming: shorter API in Python

The compiler still lowers to canonical PTO mnemonics (e.g. `trowmax`, `tmatmul`), but the Python frontend
accepts shorter spellings:

- `pto.rowmax(...)` → `pto.trowmax ...`
- `pto.matmul(...)` → `pto.tmatmul ...`

The old spellings (`pto.trowmax`, `pto.tmatmul`) remain supported.

## 3. Control flow (loops + if/else)

Supported:

- `for i in range(start, stop, step): ...` (all args must be constants or simple scalar names)
- `if a < b: ... else: ...` where the condition is a **single comparison**

Unsupported (examples):

- `while ...`
- complex boolean conditions (`and`, `or`, chained compares)
- function calls in conditions (except the small set of compile-time helpers like `sqrt(...)`)

## 4. SPMD (multi-block) programming

Inside kernels you can use:

- `bid = pto.get_block_idx()`
- `bn  = pto.get_block_num()`

Typical pattern:

```python
bn = pto.get_block_num()
bid = pto.get_block_idx()
rows_per_blk = 256 // bn
r0 = bid * rows_per_blk
r1 = r0 + rows_per_blk
for r in range(r0, r1, 16):
    ...
```

Regression runner supports per-case `block_dim` overrides (see `kernels/python/run_regression.py`).

## 5. Synchronization (events)

Python kernels should **not** manually insert `record_event` / `wait_event`.

Use `ptoas --enable-insert-sync` (enabled by default in `kernels/python/run_regression.py` and `ptoas/tools/python_kernel_e2e.py`).
If a case deadlocks on NPU, rerun it in simulator mode and inspect:

- `event_summary.txt` / `event_summary.json`
- `set_wait_snippet.txt`
- simulator logs under `camodel_logs/`

## 6. Running kernels end-to-end

Single kernel (CPU reference + NPU or simulator run):

```bash
python3 ptoas/tools/python_kernel_e2e.py kernels/python/softmax16.py --run-mode npu --timeout-sec 300
python3 ptoas/tools/python_kernel_e2e.py kernels/python/softmax16.py --run-mode sim --soc a3 --timeout-sec 600
```

Regression suite:

```bash
python3 kernels/python/run_regression.py --run-mode npu --timeout-sec 300
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --timeout-sec 600
```

## 6.1 Split kernels (CUBE + VEC stages)

Some real workloads (e.g. FlashAttention) want **cube** stages (`matmul`) and **vec** stages (`softmax`) in one logical program.
For this, the Python frontend can split one program into **multiple MLIR functions** (one per stage), and `ptoas` emits multiple device kernels.

Authoring pattern:

- Insert stage markers in Python: `pto.stage_<name>()`.
- Use a suffix convention for core selection:
  - stage name contains `cube` → compiled as `--cce-aicore-arch=<base>-cube`
  - stage name contains `vec`  → compiled as `--cce-aicore-arch=<base>-vec` and emitted with `__attribute__((aiv))`

Build/run:

- `build_fatobj_so_from_cce(...)` builds:
  - one stage fatobj `.so` per kernel (`*.stage0.so`, `*.stage1.so`, ...)
  - a tiny host wrapper `.so` that chains them in order

Example kernel:

- `kernels/python/flash_attention64_split.py`

Run it:

```bash
python3 kernels/python/run_regression.py --run-mode sim --soc a3 --cases flash_attention64_split --timeout-sec 600
python3 kernels/python/run_regression.py --run-mode npu --cases flash_attention64_split --timeout-sec 300
```

Debug helpers:

- `PTOAS_SPLIT_TRACE=1` prints the stage `.so` names as they are chained.
- Simulator artifacts go under `.../camodel_logs/` for the case.

## 7. PTO instruction coverage in Python

`pto_as.PTO` exposes a curated set of PTO ISA mnemonics (see `_PTO_KNOWN_OPS` in `binding/python/pto_as/__init__.py`), and
`ptoas.python.dsl` also exposes a matching set of stub functions (see `_PTO_ISA_OPS` in `binding/python/ptoas/python/dsl.py`).

If you add a new ISA instruction in C++ (`include/pto/common/pto_instr.hpp`), update these lists so the Python
frontend stays in sync.
