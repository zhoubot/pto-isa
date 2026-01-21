# TCOLMAX

## Introduction

Reduce each column by taking the maximum across rows.

## Math Interpretation

Let `R = src.GetValidRow()` and `C = src.GetValidCol()`. For `0 <= j < C`:

$$ \mathrm{dst}_{0,j} = \max_{0 \le i < R} \mathrm{src}_{i,j} $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tcolmax %dst, %src : (!pto.tile<...>, !pto.tile<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataOut, typename TileDataIn, typename... WaitEvents>
PTO_INST RecordEvent TCOLMAX(TileDataOut& dst, TileDataIn& src, WaitEvents&... events);
```

## Constraints

Implementation checks (NPU):

- Tile location: `dst` and `src` must be `TileType::Vec`.
- Tile layout: both tiles must be ND fractal (`isRowMajor` and `SLayout::NoneBox`).
- Data types:
  - A2A3: `half`, `float`, `int16_t`, `int32_t`.
  - A5: `half`, `float`, `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `int32_t`, `uint32_t`, `bfloat16_t`.
- DType consistency: `dst.DType == src.DType`.
- Runtime valid checks:
  - `src.GetValidCol() == dst.GetValidCol()`.
  - If `src.GetValidRow() == 0` or `src.GetValidCol() == 0`, the implementation returns early.

## Examples

### Auto

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_auto() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TCOLMAX(dst, src);
}
```

### Manual

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example_manual() {
  using SrcT = Tile<TileType::Vec, float, 16, 16>;
  using DstT = Tile<TileType::Vec, float, 1, 16>;
  SrcT src;
  DstT dst;
  TASSIGN(src, 0x1000);
  TASSIGN(dst, 0x2000);
  TCOLMAX(dst, src);
}
```
