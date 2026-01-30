# TCMP

## Introduction

Compare two tiles and write a packed predicate mask.

## Math Interpretation

Conceptually, for each element `(i, j)` in the valid region, define a predicate:

$$ p_{i,j} = \left(\mathrm{src0}_{i,j}\ \mathrm{cmpMode}\ \mathrm{src1}_{i,j}\right) $$

The predicate mask is stored in `dst` using an implementation-defined packed layout.

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tcmp %dst, %src0, %src1 {cmpMode = #pto.cmp<EQ>} : (!pto.tile<...>, !pto.tile<...>)
```
## IR Syntax

### IR-level1 (SSA)

```mlir
%dst = pto.tcmp %src0, %src1 {cmpMode = #pto.cmp<xx>} : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>
```

### IR-level2 (DPS)

```mlir
pto.tcmp ins(%src0, %src1  {cmpMode = #pto.cmp<xx>} : !pto.tile_buf<...>, !pto.tile_buf<...>) outs(%dst : !pto.tile_buf<...>)
```

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp` and `include/pto/common/type.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc, typename... WaitEvents>
PTO_INST RecordEvent TCMP(TileDataDst& dst, TileDataSrc& src0, TileDataSrc& src1, CmpMode cmpMode,
                          WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `src0/src1/dst` tile location must be `TileType::Vec`.
  - Static valid bounds: `TileDataSrc::ValidRow <= TileDataSrc::Rows` and `TileDataSrc::ValidCol <= TileDataSrc::Cols`.
  - Runtime: `src0.GetValidRow() == dst.GetValidRow()` and `src0.GetValidCol() == dst.GetValidCol()`.
  - Note: `src1` shape/valid is not validated by explicit runtime assertions in this implementation.
  - For `TileDataSrc::DType == int32_t`, the implementation uses the `EQ` compare path regardless of `cmpMode`.
- **Implementation checks (A5)**:
  - Implemented (see `include/pto/npu/a5/TCmp.hpp`).
  - The A5 implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` as the iteration domain and writes a packed predicate mask into `dst` (target-defined packing).
- **Mask encoding**:
  - The mask tile is interpreted as packed predicate bits in a target-defined layout.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 16>;
  SrcT src0, src1;
  MaskT mask;
  TCMP(mask, src0, src1, CmpMode::GT);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using MaskT = Tile<TileType::Vec, uint8_t, 16, 16>;
  SrcT src0, src1;
  MaskT mask;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(mask, 0x3000);
  TCMP(mask, src0, src1, CmpMode::GT);
}
```
