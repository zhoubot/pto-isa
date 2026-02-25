# GEMM (Python → PTO-AS → ptoas) Example

This is a minimal GEMM example written in Python that generates PTO-AS text, then uses the `ptoas` toolchain to build and run on:

- **CPU simulator** (`bin/ptoas` → C++ → `clang++ -D__CPU_SIM` → `.so`)
- **Real NPU** (`bin/ptoas` → CCE C++ → `bisheng` → fatobj `.so`)

It is intentionally small (16×16) and focuses on the end-to-end flow.

## Prereqs

- Ensure you have a working `bin/ptoas` (built from `~/llvm-project`):

```bash
ninja -C ~/llvm-project/build-mlir ptoas
cp ~/llvm-project/build-mlir/bin/ptoas ./bin/ptoas
```

- For NPU: source Ascend env and ensure `ASCEND_HOME_PATH` is set:

```bash
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
```

## Run

CPU:

```bash
python3 kernels/custom/gemm_python/run.py --target cpu
```

Or:

```bash
cd kernels/custom/gemm_python
./run.sh cpu
```

NPU:

```bash
python3 kernels/custom/gemm_python/run.py --target npu --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

Both:

```bash
python3 kernels/custom/gemm_python/run.py --target both --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```
