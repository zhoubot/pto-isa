# GEMM（Python → PTO-AS → ptoas）示例

本示例用 Python 生成 PTO-AS 文本，然后通过 `ptoas` 工具链在以下平台跑通并用 numpy 校验：

- **CPU 仿真**（`bin/ptoas` → 生成 C++ → `clang++ -D__CPU_SIM` → `.so`）
- **真实 NPU**（`bin/ptoas` → 生成 CCE C++ → `bisheng` → fatobj `.so`）

示例规模为 16×16，目的是验证端到端链路。

## 前置条件

- 确保你有可用的 `bin/ptoas`（从 `~/llvm-project` 构建并拷贝到本仓库）：

```bash
ninja -C ~/llvm-project/build-mlir ptoas
cp ~/llvm-project/build-mlir/bin/ptoas ./bin/ptoas
```

- NPU 环境：source Ascend 环境并设置 `ASCEND_HOME_PATH`：

```bash
source $HOME/Ascend/ascend-toolkit/latest/bin/setenv.bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
```

## 运行

CPU：

```bash
python3 kernels/custom/gemm_python/run.py --target cpu
```

或者：

```bash
cd kernels/custom/gemm_python
./run.sh cpu
```

NPU：

```bash
python3 kernels/custom/gemm_python/run.py --target npu --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```

两者都跑：

```bash
python3 kernels/custom/gemm_python/run.py --target both --ascend-home "$ASCEND_HOME_PATH" --device 0 --block-dim 1
```
