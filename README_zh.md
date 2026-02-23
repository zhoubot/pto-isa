<p align="center">
  <img src="docs/figures/pto_logo.svg" alt="PTO Tile Lib" width="220" />
</p>

# PTO Tile Library

面向昇腾平台的 **tile 级高性能算子/指令实现库**，基于 **PTO（Parallel Tile Operation）虚拟 ISA**。

- **文档入口**：`docs/`（建议从 [`docs/README_zh.md`](docs/README_zh.md) 开始）
- **快速上手**（Windows/Linux/macOS）：[`docs/getting-started_zh.md`](docs/getting-started_zh.md)
- **ISA / 头文件说明**：[`include/README_zh.md`](include/README_zh.md)
- **English**: [`README.md`](README.md)

## 新闻

- **2025-12-27**：PTO Tile Library 正式开源发布。

## PTO 是什么？

昇腾硬件架构会随代际演进发生明显变化，底层指令集也会随之调整。**PTO** 通过将抽象提升到 tile 级别，提供更稳定的虚拟 ISA：

- 定义 **90+** 条标准 tile 指令（ISA 定义）
- 在不同代际之间提供更好的 **兼容性/可迁移性**
- 同时保留足够的 **性能调优空间**（tile size/shape、指令顺序、流水线组织等）

本仓库实现了 PTO 指令的一个持续增长子集，并提供性能内核、CPU 仿真、测试与配套文档。

## 目标用户

PTO Tile Lib 并不面向入门级用户，主要面向：

- 直接对接昇腾硬件的框架后端开发者
- 跨平台应用开发者（需要适配多代昇腾）
- 高性能算子/内核开发者（手工实现）

## 集成情况

PTO 指令已集成到：

- [PyPTO](https://gitcode.com/cann/pypto/)
- [TileLang Ascend](https://github.com/tile-ai/tilelang-ascend/)
- 更多语言/前端持续完善中

## 平台支持

- Ascend A2（Ascend 910B）
- Ascend A3（Ascend 910C）
- Ascend A5（Ascend 950）
- CPU 仿真（x86_64 / AArch64）

更多细节请参考：[`include/README_zh.md`](include/README_zh.md)

## 环境依赖

### CPU 仿真

- C++ 编译器（Clang/GCC/MSVC）
- CMake
- Python 3

### 昇腾（NPU / simulator）

- Ascend CANN Toolkit **>= 8.3**（见 `version.info`）
- 已正确配置运行环境（需要 source `setenv.bash`）

## 快速开始（CPU 仿真，推荐第一步）

CPU 仿真跨平台，不依赖昇腾驱动/CANN。

执行完整 CPU 流程：

```bash
python3 tests/run_cpu.py --clean --verbose
```

运行 demo（可选）：

```bash
python3 tests/run_cpu.py --demo gemm --verbose
python3 tests/run_cpu.py --demo flash_attn --verbose
```

运行 CPU 仿真测试：

```bash
chmod +x ./tests/run_cpu_tests.sh
./tests/run_cpu_tests.sh
```

## 快速开始（昇腾 ST）

> ST 测试需要可用的昇腾 CANN 环境，通常仅在 Linux 上使用。

1）配置 CANN 环境变量：

```bash
# root 安装
source /usr/local/Ascend/cann/bin/setenv.bash

# 非 root 用户安装
source $HOME/Ascend/cann/bin/setenv.bash
```

2）运行推荐测试集：

```bash
chmod +x ./tests/run_st.sh
./tests/run_st.sh a5 npu simple
./tests/run_st.sh a3 sim all
```

3）运行单个 ST 用例：

```bash
python3 tests/script/run_st.py -r [sim|npu] -v [a3|a5] -t [TEST_CASE] -g [GTEST_FILTER_CASE]

# 示例
python3 tests/script/run_st.py -r npu -v a3 -t tmatmul -g TMATMULTest.case1
python3 tests/script/run_st.py -r sim -v a5 -t tmatmul -g TMATMULTest.case1
```

说明：`a3` 后端覆盖 A2/A3 系列（`include/pto/npu/a2a3`）。

## 构建 / 打包

一键构建与运行：

```bash
chmod +x build.sh

# 运行完整 ST 测试
./build.sh --run_all --a3 --sim

# 运行精简 ST 测试
./build.sh --run_simple --a5 --npu

# 打包
./build.sh --pkg
```

## 文档站点（MkDocs）

`docs/mkdocs/` 下提供 MkDocs（Read the Docs 主题）站点：

```bash
python -m pip install -r docs/mkdocs/requirements.txt
python -m mkdocs serve -f docs/mkdocs/mkdocs.yml
```

构建静态站点：

```bash
python -m mkdocs build -f docs/mkdocs/mkdocs.yml
```

通过 CMake 构建文档：

```bash
cmake -S docs -B build/docs -DPython3_EXECUTABLE=$PWD/.venv-mkdocs/bin/python
cmake --build build/docs --target pto_docs
```

## 性能参考

本仓库包含面向性能的 kernels，并给出参考测量数据与可复现的实验设置。

### GEMM（A2/A3 参考）

- Kernel：`kernels/manual/a2a3/gemm_performance/`
- 调参与分析说明：`kernels/manual/a2a3/gemm_performance/README_zh.md`

在 Ascend A3（24 核）上测量（fp16 输入 → fp32 输出）：

| 参数 | TMATMUL（Cube）占比 | TEXTRACT 占比 | TLOAD 占比 | TSTORE 占比 | 执行时间（ms） |
| --- | --- | --- | --- | --- | --- |
| `m=1536` `k=1536` `n=1536` | 54.5% | 42.2% | 72.2% | 7.7% | 0.0388 |
| `m=3072` `k=3072` `n=3072` | 79.0% | 62.0% | 90.9% | 5.8% | 0.2067 |
| `m=6144` `k=6144` `n=6144` | 86.7% | 68.1% | 95.2% |  3.1% | 1.5060 |
| `m=7680` `k=7680` `n=7680` | 80.6% | 63.0% | 98.4% |  2.4% | 3.1680 |

![GEMM 性能参考（Ascend A3，24 核）](docs/figures/performance/gemm_performance_a3.svg)

### Flash Attention（A2/A3 参考）

- Kernel：`kernels/manual/a2a3/flash_atten/`
- 调参与分析说明：`kernels/manual/a2a3/flash_atten/README_zh.md`

![Flash Attention 归一化 TFLOPS（A2/A3）](docs/figures/performance/fa_normalized_tflops_a2a3.svg)

## 路线图（Roadmap）

| 功能 | 描述 | 范围 |
| --- | --- | --- |
| PTO Auto Mode | BiSheng 编译器支持：自动分配 tile buffer 并插入同步。 | 编译器 / 工具链 |
| PTO Tile Fusion | BiSheng 编译器支持：自动融合 tile 操作。 | 编译器 / 工具链 |
| PTO-AS | PTO ISA 的字节码（Byte Code）支持。 | 编译器 / 工具链 |
| 卷积扩展 | PTO ISA 对卷积 kernel 的支持。 | ISA 扩展 |
| 集合通信扩展 | PTO ISA 对集合通信 kernel 的支持。 | ISA 扩展 |
| 系统调度扩展 | PTO ISA 对 SPMD/MPMD 编程的调度支持。 | ISA 扩展 |

## 仓库结构

- `include/`：PTO C++ 头文件（见 [`include/README_zh.md`](include/README_zh.md)）
- `kernels/`：自定义算子与 kernel 实现（见 [`kernels/README_zh.md`](kernels/README_zh.md)）
- `docs/`：ISA 指令、API 指南与示例（见 [`docs/README_zh.md`](docs/README_zh.md)）
- `tests/`：ST/CPU 测试脚本与用例（见 [`tests/README_zh.md`](tests/README_zh.md)）
- `scripts/`：打包与发布脚本（见 [`scripts/README_zh.md`](scripts/README_zh.md)）
- `build.sh`、`tests/run_st.sh`：构建、打包与示例运行入口

## 贡献指南

见：[`CONTRIBUTING.md`](CONTRIBUTING.md)

## 安全

见：[`SECURITY.md`](SECURITY.md)

## 许可证

本项目基于 CANN Open Software License Agreement Version 2.0 进行许可。详情见根目录下的 [`LICENSE`](LICENSE) 文件。
