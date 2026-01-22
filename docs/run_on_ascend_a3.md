# Run On Ascend 910B (A3)

This repo can generate **Ascend A2/A3** (A3 = Ascend 910B) kernel `.cpp` files that use the vendored PTO header backend in `include/pto/`.

This repo also provides a small **ACL smoke runner** so you can compile and execute the generated kernels directly on this machine.

## Prerequisites (this machine)

- Ascend toolkit installed and environment set.
  - This machine already has `ccec` and `bisheng` under `$ASCEND_HOME_PATH`.

Quick check:

```bash
echo "$ASCEND_HOME_PATH"
which ccec
which bisheng
```

## Option A (recommended): build + run all examples on NPU

```bash
chmod +x scripts/a3/build_and_run_examples.sh
scripts/a3/build_and_run_examples.sh
```

Optional toggles:

- Skip re-generating outputs: `PTO_SKIP_GENERATE=1 scripts/a3/build_and_run_examples.sh`
- Skip the LLaMA demo: `PTO_SKIP_LLAMA=1 scripts/a3/build_and_run_examples.sh`

The runner prints a per-output checksum so you can confirm kernels executed on-device.

## Option A2: simulator + CPU compare (single kernel)

This runs the same kernel on CPU and A3 simulator, then compares output checksums.

```bash
python scripts/a3/generate_smoke_ops.py
chmod +x scripts/a3/smoke_one_compare.sh
scripts/a3/smoke_one_compare.sh sim smoke_ops op_add.cpp
```

Notes:

- The simulator is very slow; keep tests small and targeted.
- The compare script matches checksums per output memref; if the simulator logs unsupported ISA errors, this will still fail.
- For simulator runs, the default SoC is `Ascend910B1`. Override with `PTO_SOC_VERSION` if needed.

## Option B: generate + compile + run one kernel

### 1) Generate an A3 kernel `.cpp`

Example: generate `rowmax` for the fused-softmax demo:

```bash
python pto_compile.py codegen \
  --entry examples.pto_fused_softmax:create_rowmax_func \
  --backend ascend_a2a3 \
  --output-base-dir examples \
  --output-prefix fused_softmax
```

Output:

- `examples/output_ascend_a2a3/fused_softmax/rowmax.cpp`
- `examples/output_pto/fused_softmax/rowmax.pto`

### 2) Compile the kernel to a `.so` (CCE mode)

```bash
ccec -xcce --cce-aicore-arch=dav-c220-vec -O2 -std=c++17 \
  -DMEMORY_BASE -DPTO_NPU_SMOKE_RUNNER \
  -fPIC -c \
  -Iinclude \
  -I"$ASCEND_HOME_PATH/include" \
  -I"$ASCEND_HOME_PATH/pkg_inc" \
  -I"$ASCEND_HOME_PATH/pkg_inc/profiling" \
  -I"$ASCEND_HOME_PATH/pkg_inc/runtime/runtime" \
  examples/output_ascend_a2a3/fused_softmax/rowmax.cpp \
  -o build/a3/rowmax.o

ccec -shared build/a3/rowmax.o -o build/a3/rowmax.so --cce-fatobj-link
```

Notes:

- `--cce-aicore-arch=dav-c220-vec` is the typical setting for vector-kernel testcases on A3/910B. For cube/matmul kernels, your harness may use `dav-c220-cube`.
- The `.cpp` contains `pto_launch(...)` and metadata symbols when `PTO_NPU_SMOKE_RUNNER` is enabled.

### 3) Run it (ACL runner)

```bash
build/a3/pto_npu_runner build/a3/rowmax.so
```

## Option B2: run a vadd kernel (op_add) on NPU

`op_add` is a simple vector add and maps to the same flow as the baseline vadd demo.

```bash
python scripts/a3/generate_smoke_ops.py
chmod +x scripts/a3/smoke_one_compare.sh
scripts/a3/smoke_one_compare.sh npu smoke_ops op_add.cpp
```

## Option C: run examples one by one (CPU / sim / NPU)

This iterates output subdirectories and runs them one-by-one.

```bash
python scripts/run_examples.py --mode npu
python scripts/run_examples.py --mode sim --skip-llama
python scripts/run_examples.py --mode cpu
```

Helpful flags:

- `--subdirs fused_softmax,torch_nn` to narrow the run.
- `--skip-generate` to reuse existing outputs.
- `--soc-version Ascend910B1` to pin the simulator target.

## Option D: run single kernel on NPU + CPU compare

```bash
python scripts/a3/generate_smoke_ops.py
chmod +x scripts/a3/smoke_one_compare.sh
scripts/a3/smoke_one_compare.sh npu smoke_ops op_add.cpp
```

## A5

Generate A5 kernels by switching the backend:

```bash
python pto_compile.py codegen \
  --entry examples.pto_fused_softmax:create_rowmax_func \
  --backend ascend_a5 \
  --output-base-dir examples \
  --output-prefix fused_softmax
```

This writes to `examples/output_ascend_a5/...`.

## From .pto to .cpp (direct)

If you already have a `.pto` file, you can generate backend code directly:

```bash
python scripts/a3/codegen_from_pto.py \
  --pto examples/output_pto/fused_softmax/rowmax.pto \
  --backend ascend_a2a3 \
  --output-prefix fused_softmax
```

Then compile + run it:

```bash
chmod +x scripts/a3/smoke_one.sh
scripts/a3/smoke_one.sh npu fused_softmax rowmax.cpp
```
