# include/

PTO Tile Lib 对外的 C/C++ 头文件（以模板化、基本 header-only 为主）。上层框架/算子可以通过这些头文件生成 PTO ISA 的 Tile 指令序列。

## 快速开始

推荐直接 include 统一入口头：

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` 会根据构建配置选择合适的后端（CPU 仿真/stub 或 NPU 实现）。详情见 `include/pto/README.md`。

## 目录结构

- `include/pto/`：公共 PTO ISA API 与后端实现（common / cpu / npu）

## 相关文档

- ISA 指南：`docs/README.md`
- 入门指南：`docs/getting-started.md`

## PTO 指令实现状态（CPU / A2 / A3 / A5）

下表用于跟踪每条指令在不同后端的可用性：

- **CPU**：`__CPU_SIM`（CPU 仿真后端）。
- **A2（Ascend 910B）/ A3（Ascend 910C）**：当前共享 `include/pto/npu/a2a3/` 的实现（因此两列状态相同）。
- **A5（Ascend 950）**：使用 `include/pto/npu/a5/` 的实现。
- **TODO**：表示该指令属于公共 API，但对应后端实现尚不可用。

| 指令 | CPU | A2 | A3 | A5 |
|---|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER.md) | 是 | TODO | TODO | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | 是 | TODO | TODO | TODO |
| [`TABS`](../docs/isa/TABS.md) | TODO | TODO | TODO | TODO |
| [`TADD`](../docs/isa/TADD.md) | 是 | 是 | 是 | 是 |
| [`TADDC`](../docs/isa/TADDC.md) | 是 | TODO | TODO | TODO |
| [`TADDS`](../docs/isa/TADDS.md) | 是 | 是 | 是 | 是 |
| [`TADDSC`](../docs/isa/TADDSC.md) | 是 | TODO | TODO | TODO |
| [`TAND`](../docs/isa/TAND.md) | 是 | TODO | TODO | TODO |
| [`TANDS`](../docs/isa/TANDS.md) | 是 | TODO | TODO | TODO |
| [`TASSIGN`](../docs/isa/TASSIGN.md) | 是 | 是 | 是 | 是 |
| [`TCI`](../docs/isa/TCI.md) | 是 | 是 | 是 | 是 |
| [`TCMP`](../docs/isa/TCMP.md) | TODO | TODO | TODO | TODO |
| [`TCMPS`](../docs/isa/TCMPS.md) | 是 | 是 | 是 | 是 |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | 是 | TODO | TODO | TODO |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | 是 | 是 | 是 | 是 |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | TODO | TODO | TODO | TODO |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | 是 | 是 | 是 | 是 |
| [`TCVT`](../docs/isa/TCVT.md) | 是 | TODO | TODO | TODO |
| [`TDIV`](../docs/isa/TDIV.md) | 是 | 是 | 是 | 是 |
| [`TDIVS`](../docs/isa/TDIVS.md) | 是 | 是 | 是 | 是 |
| [`TEXP`](../docs/isa/TEXP.md) | 是 | 是 | 是 | 是 |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | 是 | 是 | 是 | 是 |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | 是 | 是 | 是 | 是 |
| [`TGATHER`](../docs/isa/TGATHER.md) | 是 | 是 | 是 | 是 |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | 是 | 是 | 是 | 是 |
| [`TLOAD`](../docs/isa/TLOAD.md) | 是 | 是 | 是 | 是 |
| [`TLOG`](../docs/isa/TLOG.md) | TODO | TODO | TODO | TODO |
| [`TLRELU`](../docs/isa/TLRELU.md) | 是 | TODO | TODO | TODO |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | 是 | 是 | 是 | 是 |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | 是 | 是 | 是 | 是 |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | 是 | 是 | 是 | 是 |
| [`TMAX`](../docs/isa/TMAX.md) | 是 | 是 | 是 | 是 |
| [`TMAXS`](../docs/isa/TMAXS.md) | 是 | TODO | TODO | TODO |
| [`TMIN`](../docs/isa/TMIN.md) | TODO | TODO | TODO | TODO |
| [`TMINS`](../docs/isa/TMINS.md) | 是 | 是 | 是 | 是 |
| [`TMOV`](../docs/isa/TMOV.md) | 是 | 是 | 是 | 是 |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | TODO | TODO | TODO | TODO |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | 是 | 是 | 是 | 是 |
| [`TMUL`](../docs/isa/TMUL.md) | 是 | 是 | 是 | 是 |
| [`TMULS`](../docs/isa/TMULS.md) | 是 | 是 | 是 | 是 |
| [`TNEG`](../docs/isa/TNEG.md) | 是 | TODO | TODO | TODO |
| [`TNOT`](../docs/isa/TNOT.md) | 是 | TODO | TODO | TODO |
| [`TOR`](../docs/isa/TOR.md) | 是 | TODO | TODO | TODO |
| [`TORS`](../docs/isa/TORS.md) | 是 | TODO | TODO | TODO |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | 是 | 是 | 是 | 是 |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | 是 | 是 | 是 | 是 |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | 是 | 是 | 是 | 是 |
| [`TPRELU`](../docs/isa/TPRELU.md) | 是 | TODO | TODO | TODO |
| [`TRECIP`](../docs/isa/TRECIP.md) | TODO | TODO | TODO | TODO |
| [`TRELU`](../docs/isa/TRELU.md) | 是 | TODO | TODO | TODO |
| [`TREM`](../docs/isa/TREM.md) | 是 | TODO | TODO | TODO |
| [`TREMS`](../docs/isa/TREMS.md) | 是 | TODO | TODO | TODO |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | TODO | TODO | TODO | TODO |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | 是 | 是 | 是 | 是 |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | 是 | 是 | 是 | TODO |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL.md) | 是 | 是 | 是 | TODO |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | 是 | 是 | 是 | TODO |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | 是 | 是 | 是 | 是 |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | TODO | TODO | TODO | TODO |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | 是 | 是 | 是 | 是 |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | 是 | 是 | 是 | 是 |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | TODO | TODO | TODO | TODO |
| [`TSEL`](../docs/isa/TSEL.md) | 是 | 是 | 是 | 是 |
| [`TSELS`](../docs/isa/TSELS.md) | TODO | TODO | TODO | TODO |
| [`TSHL`](../docs/isa/TSHL.md) | 是 | TODO | TODO | TODO |
| [`TSHR`](../docs/isa/TSHR.md) | 是 | TODO | TODO | TODO |
| [`TSORT32`](../docs/isa/TSORT32.md) | 是 | 是 | 是 | 是 |
| [`TSQRT`](../docs/isa/TSQRT.md) | 是 | 是 | 是 | 是 |
| [`TSTORE`](../docs/isa/TSTORE.md) | 是 | 是 | 是 | 是 |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | TODO | TODO | TODO | TODO |
| [`TSUB`](../docs/isa/TSUB.md) | 是 | 是 | 是 | 是 |
| [`TSUBC`](../docs/isa/TSUBC.md) | 是 | TODO | TODO | TODO |
| [`TSUBS`](../docs/isa/TSUBS.md) | 是 | TODO | TODO | TODO |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | 是 | TODO | TODO | TODO |
| [`TSYNC`](../docs/isa/TSYNC.md) | TODO | TODO | TODO | TODO |
| [`TTRANS`](../docs/isa/TTRANS.md) | 是 | 是 | 是 | 是 |
| [`TXOR`](../docs/isa/TXOR.md) | 是 | TODO | TODO | TODO |
| [`TXORS`](../docs/isa/TXORS.md) | 是 | TODO | TODO | TODO |
