# Tutorial: How the Toolchain Transforms a Kernel (GEMM256 Example)

This is a *general* “how the flow works” tutorial that shows what each stage produces.
We use `kernels/python/gemm256.py` as a concrete example, but the same pipeline applies to any kernel written with the Python frontend.

Pipeline (conceptually):

1) **Python frontend** (user code) → structured PTO-AS text (`.pto`)
2) **`ptoas`** (compiler) → Ascend CCE kernel source (`.cpp`) and optionally a compiled artifact (`.bin`)
3) **`bisheng`** (Ascend toolchain) → loadable fatobj library (`.so`)
4) **Runner** (ACL) → launch kernel on NPU (or simulator)

![GEMM pipeline](../../figures/gemm_pipeline.png)

## Prerequisites

- Run commands from the repo root.
- Python `>= 3.8`, and `numpy`.
- A built `ptoas` binary at `ptoas/mlir/build/bin/ptoas` (see `ptoas/mlir/README.md`).
- For NPU (or simulator): Ascend toolkit installed and `ASCEND_HOME_PATH` set; `bisheng` on `PATH`; Python can `import acl`.

Recommended env vars:

```bash
export ASCEND_HOME_PATH=/path/to/ascend-toolkit/latest
export PTOAS=$PWD/ptoas/mlir/build/bin/ptoas
export OUT=/tmp/pto_gemm256
```

## Stage A: The user kernel (Python frontend)

`kernels/python/gemm256.py` is intentionally “normal Python”: it uses the `pto_as` builder object to construct a program.
This is the *source* that the Python frontend compiles into PTO-AS.

Excerpt:

```python
from pto_as import PTO

def gemm256():
    pto = PTO("gemm256")
    pto.prologue()

    a = pto.tensor("a", (256, 256), dtype="f16", role="in")
    b = pto.tensor("b", (256, 256), dtype="f16", role="in")
    c = pto.tensor("c", (256, 256), dtype="f32", role="out")

    # allocate tiles, then loop in (M,N,K) tiles...
    for mi in range(0, 256, 16):
        for nj in range(0, 256, 16):
            for kk in range(0, 256, 16):
                a_mat = pto.load(a, mi, kk)
                b_mat = pto.load(b, kk, nj)
                # tmov + tmatmul_acc...
            pto.store(c, mi, nj, c_acc)

    pto.epilogue()
    return pto.program()
```

What matters here:

- You write *Python control flow* (`for`, `if`) and *tile ops* (`load`, `mov`, `tmatmul_acc`, `store`).
- The frontend captures this and emits a **PTO-AS program**.

## Stage B: Python → PTO-AS (`.pto`, no sync/events)

Generate a `.pto` from the Python file:

```bash
python3 - <<'PY'
from pathlib import Path
from ptoas.python import binding

outdir = Path(__import__("os").environ["OUT"])
outdir.mkdir(parents=True, exist_ok=True)

pto_path = outdir / "gemm256.pto"
binding.write_pto(Path("kernels/python/gemm256.py"), kernel="gemm256", out_path=pto_path, universal=True)
print("wrote:", pto_path)
PY
```

The `.pto` is structured PTO-AS. It contains:

1) A small **host spec** header (shapes/dtypes/seed) used by runners.
2) A `prologue` + tile allocations.
3) Lowered control flow (`scf.for`, `scf.if`) and tile ops (`pto.tload`, `pto.tmov`, `pto.tmatmul_acc`, `pto.tstore`).

Excerpt (what you should expect to see near the top of the generated `.pto`):

```text
; PTO_HOST_SPEC_BEGIN v1
; { "args": [ ... ], "block_dim": 1, "kernel_name": "pto_kernel", "seed": 0 }
; PTO_HOST_SPEC_END
prologue
%a = pto.make_tensor_view %arg0, dtype=f16, shape=[256,256] strides=[256,1], layout=ND
...
scf.for %mi = 0 to 256 step 16 {
  scf.for %nj = 0 to 256 step 16 {
    scf.for %kk = 0 to 256 step 16 {
      %a_mat = pto.tload %a[%mi, %kk]
      %b_mat = pto.tload %b[%kk, %nj]
      ...
      %c_acc = pto.tmatmul_acc %c_acc, %a_left_0, %b_right_0
```

For this tutorial, the `.pto` is “**without sync**” (no events / `tsync` inserted):

```bash
rg -n "tsync|event" $OUT/gemm256.pto || true
```

## Stage C: PTO-AS → CCE C++ (via `ptoas`)

Now `ptoas` takes the `.pto` and produces an Ascend CCE kernel source.

Key idea:

- The **PTO-AS program** is the “compiler IR” that still talks about tiles and PTO intrinsics.
- `ptoas` lowers that IR into compilable **CCE C++** for your target architecture (here: cube core).

Command (still “without sync” because we do *not* pass `--insert-events`):

```bash
$PTOAS $OUT/gemm256.pto \
  --target npu \
  -o $OUT/gemm256.cce.cpp \
  --kernel-name pto_kernel_gemm256 \
  --arch dav-c220-cube \
  --memory-model MEMORY_BASE \
  --repo-root $PWD \
  --ascend-home $ASCEND_HOME_PATH \
  --emit-bin=$OUT/gemm256.bin
```

Outputs:

- `$OUT/gemm256.cce.cpp`: Ascend CCE kernel source
- `$OUT/gemm256.bin`: an additional artifact emitted by `ptoas` (uses the toolkit under `$ASCEND_HOME_PATH`)

Excerpt (you should see an exported kernel entrypoint similar to this pattern):

```cpp
extern "C" __global__ AICORE void pto_kernel_gemm256(/* GM_ADDR args... */) {
  // tile allocs, tload/tmov/tmatmul_acc/tstore lowering...
}
```

Optional: if you want the compiler to *heuristically* insert synchronization/events during lowering, add:

```bash
  --insert-events
```

## Stage D: CCE C++ → loadable `.so` (fatobj)

The easiest way to run from Python is a shared library exposing a stable entrypoint `ptoas_launch(...)`.
`ptoas.python.pipeline.build_fatobj_so_from_cce` generates a tiny wrapper, compiles with `bisheng -xcce`, and links a fatobj `.so`.

```bash
python3 - <<'PY'
import os
from pathlib import Path
from ptoas.python import pipeline

ascend_home = Path(os.environ["ASCEND_HOME_PATH"])
outdir = Path(os.environ["OUT"])

pipeline.build_fatobj_so_from_cce(
    cce_path=outdir / "gemm256.cce.cpp",
    out_so=outdir / "libgemm256_npu.so",
    arch="dav-c220-cube",
    ascend_home=ascend_home,
)
print("wrote:", outdir / "libgemm256_npu.so")
PY
```

Wrapper shape (this is what gets linked into the `.so` to make launching uniform):

```cpp
extern "C" void ptoas_launch(void *stream, uint32_t blockDim, void *arg0, void *arg1, void *arg2) {
    pto_kernel_gemm256<<<blockDim, nullptr, stream>>>((GM_ADDR)arg0, (GM_ADDR)arg1, (GM_ADDR)arg2);
}
```

## Stage E: Run on NPU (or simulator)

### Recommended: end-to-end runner + correctness check

`kernels/python/run_gemm256.py` shows the intended user flow:

- compile Python → `.pto`
- compile `.pto` → CPU C++ → CPU `.so` → run CPU reference
- compile `.pto` → CCE C++ → NPU `.so` → run on device (or simulator)
- compare CPU vs NPU outputs

NPU:

```bash
python3 kernels/python/run_gemm256.py \
  --run-mode npu \
  --ascend-home $ASCEND_HOME_PATH \
  --ptoas $PTOAS \
  --outdir $OUT \
  --no-insert-events
```

Simulator:

```bash
python3 kernels/python/run_gemm256.py \
  --run-mode sim \
  --soc a5 \
  --ascend-home $ASCEND_HOME_PATH \
  --ptoas $PTOAS \
  --outdir $OUT \
  --no-insert-events
```

### Minimal: run the `.so` directly via ACL

```bash
python3 - <<'PY'
import os
from pathlib import Path
from ptoas.python import pipeline

outdir = Path(os.environ["OUT"])
pto_text = (outdir / "gemm256.pto").read_text(encoding="utf-8")
host_spec = pipeline.parse_or_default_host_spec(pto_text=pto_text)
host_arrays = pipeline.make_host_arrays(host_spec)

out = pipeline.run_npu_kernel_from_so(
    so_path=outdir / "libgemm256_npu.so",
    host_spec=host_spec,
    host_arrays=host_arrays,
    device_id=0,
    block_dim=1,
)
print("output:", out[0].shape, out[0].dtype)
PY
```

## Where the pieces live (for readers who want to dig deeper)

- Python frontend → PTO-AS: `ptoas/python/binding.py` (calls AST frontend)
- End-to-end compilation helpers: `ptoas/python/pipeline.py`
- `ptoas` tool build/run: `ptoas/mlir/README.md`

## Troubleshooting

- If `ptoas` fails to find toolkit components, double-check `ASCEND_HOME_PATH`.
- If you want to see underlying compiler commands, use `--verbose-build` (runner) and/or set `PTOAS_QUIET=0`.
- If a kernel needs explicit ordering between memory and compute (more complex pipelines), try `--insert-events`.
