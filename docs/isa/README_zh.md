<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA 参考

本目录是 PTO Tile Lib ISA 的指令参考（每条指令一页）。

- 权威来源：[`docs/reference/pto-intrinsics-header.md`](../reference/pto-intrinsics-header.md)（C++ intrinsics，声明于 `include/pto/common/pto_instr.hpp`）
- 通用约定（操作数、事件、修饰符）：[`docs/isa/conventions.md`](conventions.md)
- IR 形式化语法（IR-level1 / IR-level2）：[`docs/grammar/PTO-ASM.def`](../grammar/PTO-ASM.def)

## 逐元素（Tile × Tile）
- [`TADD`](TADD.md) — 两个 tile 的逐元素相加。
- [`TABS`](TABS.md) — tile 的逐元素绝对值。
- [`TSUB`](TSUB.md) — 两个 tile 的逐元素相减。
- [`TMUL`](TMUL.md) — 两个 tile 的逐元素相乘。
- [`TDIV`](TDIV.md) — 两个 tile 的逐元素相除。
- [`TREM`](TREM.md) — 两个 tile 的逐元素取余。
- [`TSHL`](TSHL.md) — 两个 tile 的逐元素左移。
- [`TSHR`](TSHR.md) — 两个 tile 的逐元素右移。
- [`TAND`](TAND.md) — 两个 tile 的逐元素按位与。
- [`TOR`](TOR.md) — 两个 tile 的逐元素按位或。
- [`TXOR`](TXOR.md) — 两个 tile 的逐元素按位异或。
- [`TMIN`](TMIN.md) — 两个 tile 的逐元素最小值。
- [`TMAX`](TMAX.md) — 两个 tile 的逐元素最大值。
- [`TEXP`](TEXP.md) — 逐元素指数。
- [`TLOG`](TLOG.md) — tile 的逐元素自然对数。
- [`TSQRT`](TSQRT.md) — 逐元素平方根。
- [`TRSQRT`](TRSQRT.md) — 逐元素倒平方根。
- [`TRECIP`](TRECIP.md) — 逐元素倒数。
- [`TNEG`](TNEG.md) — 逐元素取负。
- [`TNOT`](TNOT.md) — 逐元素按位取反。
- [`TRELU`](TRELU.md) — 逐元素 ReLU。
- [`TPRELU`](TPRELU.md) — 逐元素 PReLU（带每元素 slope tile 的参数化 ReLU）。
- [`TADDC`](TADDC.md) — 逐元素三元加法：`src0 + src1 + src2`。
- [`TSUBC`](TSUBC.md) — 逐元素三元运算：`src0 - src1 + src2`。
- [`TSEL`](TSEL.md) — 通过 mask tile 在两张 tile 之间逐元素选择。
- [`TCMP`](TCMP.md) — 比较两张 tile 并写入压缩的谓词 mask。
- [`TCVT`](TCVT.md) — 带指定舍入模式的逐元素类型转换。

## Tile × 标量 / 立即数
- [`TADDS`](TADDS.md) — tile 逐元素加一个标量。
- [`TSUBS`](TSUBS.md) — tile 逐元素减一个标量。
- [`TDIVS`](TDIVS.md) — 与标量的逐元素除法（tile/scalar 或 scalar/tile）。
- [`TMULS`](TMULS.md) — tile 逐元素乘一个标量。
- [`TREMS`](TREMS.md) — 与标量的逐元素取余：`fmod(src, scalar)`（或整数的 `%`）。
- [`TMAXS`](TMAXS.md) — tile 与标量逐元素取 max：`max(src, scalar)`。
- [`TMINS`](TMINS.md) — tile 与标量逐元素取 min。
- [`TANDS`](TANDS.md) — tile 与标量逐元素按位与。
- [`TORS`](TORS.md) — tile 与标量逐元素按位或。
- [`TXORS`](TXORS.md) — tile 与标量逐元素按位异或。
- [`TCMPS`](TCMPS.md) — tile 与标量比较，并写入逐元素的比较结果。
- [`TEXPANDS`](TEXPANDS.md) — 将标量广播到目标 tile。
- [`TSELS`](TSELS.md) — 使用标量 `selectMode` 在两张 tile 之间选择（全局选择）。
- [`TLRELU`](TLRELU.md) — 带标量 slope 的 Leaky ReLU。
- [`TADDSC`](TADDSC.md) — 逐元素融合加法：`src0 + scalar + src1`。
- [`TSUBSC`](TSUBSC.md) — 逐元素融合运算：`src0 - scalar + src1`。

## 轴向归约 / 展开
- [`TROWSUM`](TROWSUM.md) — 对每行按列求和。
- [`TROWMAX`](TROWMAX.md) — 对每行按列取最大值。
- [`TROWMIN`](TROWMIN.md) — 对每行按列取最小值。
- [`TROWEXPAND`](TROWEXPAND.md) — 将源 tile 每行的第一个元素广播到目标行。
- [`TROWEXPANDDIV`](TROWEXPANDDIV.md) — 行级广播除法：对 `src0` 的每一行除以每行标量向量 `src1`。
- [`TROWEXPANDMUL`](TROWEXPANDMUL.md) — 行级广播乘法：对 `src0` 的每一行乘以每行标量向量 `src1`。
- [`TROWEXPANDSUB`](TROWEXPANDSUB.md) — 行级广播减法：对 `src0` 的每一行减去每行标量向量 `src1`。
- [`TCOLSUM`](TCOLSUM.md) — 对每列按行求和。
- [`TCOLMAX`](TCOLMAX.md) — 对每列按行取最大值。
- [`TCOLMIN`](TCOLMIN.md) — 对每列按行取最小值。
- [`TCOLEXPAND`](TCOLEXPAND.md) — 将源 tile 每列的第一个元素广播到目标列。

## Padding
- [`TFILLPAD`](TFILLPAD.md) — 将源 tile 拷贝到目标 tile，并用 `TileDataDst::PadVal` 选择的编译期 pad 值填充剩余（padding）元素（例如 `PadValue::Min` / `PadValue::Max`）。

## 内存（GM <-> Tile）
- [`TLOAD`](TLOAD.md) — 从 GlobalTensor（GM）加载到 Tile。
- [`TSTORE`](TSTORE.md) — 将 Tile 存回 GlobalTensor（GM），可选原子写或量化参数。
- [`TSTORE_FP`](TSTORE_FP.md) — 将 accumulator tile 存回 GM，并使用缩放（`fp`）tile 作为向量量化参数。
- [`MGATHER`](MGATHER.md) — 使用逐元素索引从 GM gather-load 到 tile。
- [`MSCATTER`](MSCATTER.md) — 使用逐元素索引从 tile scatter-store 到 GM。

## 矩阵乘
- [`TMATMUL`](TMATMUL.md) — 矩阵乘（GEMM），产生 accumulator / 输出 tile。
- [`TMATMUL_MX`](TMATMUL_MX.md) — 带额外缩放 tiles 的矩阵乘（用于支持目标上的混合精度/量化 matmul）。
- [`TMATMUL_ACC`](TMATMUL_ACC.md) — 带 accumulator 输入的矩阵乘（融合 accumulate）。
- [`TMATMUL_BIAS`](TMATMUL_BIAS.md) — 带 bias add 的矩阵乘。

## 数据搬运 / 布局
- [`TMOV`](TMOV.md) — tile 间搬运/拷贝，可选实现定义的转换模式。
- [`TMOV_FP`](TMOV_FP.md) — 从 accumulator tile 搬运/转换到目标 tile，并使用缩放（`fp`）tile 作为向量量化参数。
- [`TTRANS`](TTRANS.md) — 转置（需要实现定义的临时 tile）。
- [`TEXTRACT`](TEXTRACT.md) — 从源 tile 提取 sub-tile。
- [`TRESHAPE`](TRESHAPE.md) — 在保持底层字节不变的前提下，将 tile 解释为另一种 tile 类型/形状。
- [`TASSIGN`](TASSIGN.md) — 将 Tile 对象绑定到实现定义的片上地址（手动 placement）。

## 复杂指令
- [`TCI`](TCI.md) — 在目标 tile 中生成连续的整数序列。
- [`TGATHER`](TGATHER.md) — 使用索引 tile 或编译期 mask 模式进行 gather/select。
- [`TGATHERB`](TGATHERB.md) — 使用字节偏移进行 gather。
- [`TSCATTER`](TSCATTER.md) — 使用逐元素行索引将源 tile 的行 scatter 到目标 tile。
- [`TSORT32`](TSORT32.md) — 对固定大小 32 元素块排序并产生索引映射。
- [`TMRGSORT`](TMRGSORT.md) — 多个已排序列表的归并排序（元素格式与布局为实现定义）。
- [`TPARTADD`](TPARTADD.md) — 部分逐元素 add（有效区域不一致时的处理为实现定义）。
- [`TPARTMAX`](TPARTMAX.md) — 部分逐元素 max（有效区域不一致时的处理为实现定义）。
- [`TPARTMIN`](TPARTMIN.md) — 部分逐元素 min（有效区域不一致时的处理为实现定义）。

## 同步
- [`TSYNC`](TSYNC.md) — PTO 执行同步（等待事件或插入 per-op 的 pipeline barrier）。
