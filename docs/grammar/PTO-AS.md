# PTO-AS (PTO Assembly) Specification

PTO-AS is a textual, instruction-centric assembly format for PTO Tile Lib. It is designed to be:

- close to the PTO instruction set (`TADD`, `TLOAD`, `TMATMUL`, ...),
- readable and easy to diff (one instruction per line),
- compatible with MLIR tooling (MLIR-like value naming and type spellings; can be modelled as an MLIR dialect).

PTO-AS is designed to be consumed/produced by an MLIR-based assembler/disassembler.

## 1. High-Level Form

A PTO-AS program is a list of statements. The most common statement is an instruction:

```text
tadd %dst, %src0, %src1 : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>)
```

PTO-AS uses a **destination-passing style (DPS)** surface syntax: instructions explicitly name their destination
operands (typically the first operand) instead of binding SSA results with `%dst = ...`.

PTO-AS is a synchronous, line-ordered format: there is no `wait(...)` clause and no implicit event result. If a program
needs to model an explicit dependency, it uses an explicit instruction (for example `tsync`) with event operands.

Operands may also include indexed forms (commonly used by memory ops):

```text
tload %t0, %sv[%c0, %c1] : (!pto.tile<...>, !pto.tensor<...>, index, index)
```

Type signatures (`: ...`) are recommended for readability but may be omitted when the types are unambiguous in context.

## 2. Types

PTO-AS uses MLIR-like type spellings and maps them to the public C++ template types in `include/pto/*`.

### 2.1 Tile values: `!pto.tile<...>`

`!pto.tile<...>` corresponds to `pto::Tile<...>` from `include/pto/common/pto_tile.hpp` and the enums in:

- `include/pto/common/memory.hpp` (`TileType`, `BLayout`, `SLayout`)
- `include/pto/common/constants.hpp` (`PadValue`)

Canonical spelling:

```text
!pto.tile<
  loc=Vec,
  dtype=f16,
  rows=16, cols=16,
  blayout=RowMajor,
  valid=16x16,
  slayout=NoneBox,
  fractal=512,
  pad=Null
>
```

Mapping to C++:

```text
pto::Tile<
  pto::TileType::Vec, half, 16, 16,
  pto::BLayout::RowMajor, 16, 16,
  pto::SLayout::NoneBox, 512,
  pto::PadValue::Null
>
```

Notes:

- `valid=<rowValid>x<colValid>` maps to `RowValid_` / `ColValid_` (use `dyn` to represent `pto::DYNAMIC`).
- `fractal` is in **bytes** (matches `SFractalSize_`).

### 2.2 Global memory / views: `!pto.tensor<...>`

`!pto.tensor<...>` corresponds to `pto::GlobalTensor<Element_, Shape_, Stride_, Layout_>` from
`include/pto/common/pto_tile.hpp` and `pto::Layout`.

Canonical spelling:

```text
!pto.tensor<
  dtype=f16,
  shape=[1,1,1,16,16],
  stride=[1,1,1,16,1],
  layout=ND
>
```

Mapping to C++:

```text
pto::GlobalTensor<
  half,
  pto::Shape<1, 1, 1, 16, 16>,
  pto::Stride<1, 1, 1, 16, 1>,
  pto::Layout::ND
>
```

Notes:

- `shape` / `stride` are 5-D (use `dyn` to represent `pto::DYNAMIC` in any dimension).
- `layout` names follow the `pto::Layout` enum in `include/pto/common/pto_tile.hpp` (e.g. `ND`, `DN`, `NZ`, `MX_A_ND`, ...).

### 2.3 Events: `!pto.event<...>`

PTO-AS can model explicit dependencies using event-typed values.

Canonical spelling:

```text
!pto.event<src=#pto.op<TLOAD>, dst=#pto.op<TADD>>
```

This corresponds to the backend event template (for NPU targets) such as:

```text
pto::Event<pto::Op::TLOAD, pto::Op::TADD, ...>
```

### 2.4 Scalars

Scalars use MLIR builtin types like `index`, `i32`, `f32`.

## 3. Attributes

Instruction modifiers that are not positional operands (e.g., compare modes) are written as an MLIR-style attribute
dictionary:

```text
tcmp %mask, %a, %b {cmpMode = #pto.cmp<GT>} : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>)
```

## 4. Directives

PTO-AS supports a small set of non-instruction directives for declaring external inputs and constants.

Argument declaration (introduces a named value):

```text
.arg %a : !pto.tile<...>;
```

Event arguments (when modeling a dependency explicitly):

```text
.arg %e0 : !pto.event<...>;
```

Constant declaration (introduces a named value):

```text
.const %c0 = 0 : index;
```

## 5. Grammar

The normative grammar is provided in:

- `docs/grammar/PTO-AS.bnf`
