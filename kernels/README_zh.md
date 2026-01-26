# Kernels

本目录包含可运行的 kernel / operator 示例。

## 从哪里开始

- Python kernels（推荐）：`kernels/python/`
  - 流程：Python -> PTO-AS -> `ptoas` -> 构建/运行（CPU 参考实现；可选 Ascend NPU）
  - 入口脚本：`kernels/python/run_regression.py`
- 自定义示例：`kernels/custom/`

## 备注

- 本 public repo 刻意不包含 private repo 中体积较大的 `kernels/manual/` 目录树。
- 跑 NPU 需要配置 `ASCEND_HOME_PATH`（例如 `$HOME/Ascend/ascend-toolkit/latest`），并使用 `bin/ptoas`。
