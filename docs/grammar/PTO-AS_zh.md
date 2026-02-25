# PTO-AS（PTO 汇编）规范

PTO-AS 是 PTO Tile Lib 的一种面向指令的文本汇编格式，设计目标是：

- 尽量贴近 PTO 指令集（`TADD`、`TLOAD`、`TMATMUL` 等）。
- 可读、易于 diff（通常一行一条指令）。
- 兼容 MLIR 工具链（SSA 命名、类似 MLIR 的类型拼写，并以 MLIR bytecode 作为交换格式）。

PTO-AS 预期由基于 MLIR 的 assembler/disassembler 生成与消费。

## 1. 高层形式

一个 PTO-AS 程序由一系列语句组成，其中最常见的语句是指令：

```text
%dst = tadd %src0, %src1 : (!pto.tile<32x32xf32>, !pto.tile<32x32xf32>) -> !pto.tile<32x32xf32>;
```

PTO-AS 使用类似 SSA 的值名（`%dst`、`%src0`），以保持与 MLIR 汇编约定一致；这能使格式更确定（deterministic），并更容易与 MLIR bytecode 做往返（round-trip）。

PTO-AS 是一种同步、按行顺序的格式：它不提供 `wait(...)` 子句，也不隐式生成事件结果。如果程序需要显式描述依赖关系，应使用显式指令（例如 `tsync`）并通过事件操作数表达依赖。

操作数也可以包含索引形式（常见于内存指令）：

```text
%t0 = tload %sv[%c0, %c1] : (!pto.memref<...>, index, index) -> !pto.tile<...>;
```

类型签名（`: ...`）推荐保留以增强可读性；当上下文足够明确时也可省略。

## 2. 类型

PTO-AS 使用类似 MLIR 的类型拼写：

- Tile 值：`!pto.tile<...>`（不透明）
- 全局内存 / 视图：`!pto.memref<...>`（不透明）
- 事件：`!pto.event`（不透明）
- 标量：MLIR 内建类型，例如 `index`、`i32`、`f32`

Assembler 将这些类型视为**不透明**类型：它们会被携带到 bytecode 中，但不会进行语义校验，除非未来引入目标相关的 verifier。

## 3. 属性

非位置操作数的指令修饰符（例如比较模式）使用 MLIR 风格的属性字典表示：

```text
%mask = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

## 4. 指令外指令（Directives）

PTO-AS 支持一小组非指令 directive，用于声明外部输入与常量。

参数声明（引入一个 SSA 值）：

```text
.arg %a : !pto.tile<16x16xf16>;
```

事件参数（需要显式建模依赖关系时）：

```text
.arg %e0 : !pto.event;
```

常量声明（引入一个 SSA 值）：

```text
.const %c0 = 0 : index;
```

## 5. 语法

规范性的语法定义位于：

- `docs/grammar/PTO-AS.bnf`

