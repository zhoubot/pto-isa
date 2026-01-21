# TSELS

## Introduction

Select one of two source tiles using a scalar `selectMode` (global select).

For per-element selection, use `TSEL`.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$
\mathrm{dst}_{i,j} =
\begin{cases}
\mathrm{src0}_{i,j} & \text{if } \mathrm{selectMode} = 1 \\
\mathrm{src1}_{i,j} & \text{otherwise}
\end{cases}
$$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tsels %dst, %src0, %src1, %selectMode : (!pto.tile<...>, !pto.tile<...>, !pto.tile<...>, !pto.tile<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TSELS(TileData& dst, TileData& src0, TileData& src1, uint8_t selectMode, WaitEvents&... events);
```

## Constraints

- **Implementation checks (A2A3)**:
  - `TileData::DType` must be one of: `half`, `float16_t`, `float`, `float32_t`.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: the implementation expects `src0/src1/dst` to have matching valid rows/cols.
- **Implementation checks (A5)**:
  - `sizeof(TileData::DType)` must be `1`, `2`, or `4` bytes.
  - Tile location must be vector (`TileData::Loc == TileType::Vec`).
  - Static valid bounds: `TileData::ValidRow <= TileData::Rows` and `TileData::ValidCol <= TileData::Cols`.
  - Runtime: the implementation expects `src0/src1/dst` to have matching valid rows/cols.
  - Padding behavior depends on `TileData::PadVal` (`Null`/`Zero` vs `-INF/+INF` modes).
- **Valid region**:
  - The implementation uses `dst.GetValidRow()` / `dst.GetValidCol()` as the selection domain.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TSELS(dst, src0, src1, /*selectMode=*/1);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT src0, src1, dst;
  TASSIGN(src0, 0x1000);
  TASSIGN(src1, 0x2000);
  TASSIGN(dst,  0x3000);
  TSELS(dst, src0, src1, /*selectMode=*/1);
}
```
