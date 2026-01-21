# TLRELU

## Introduction

Leaky ReLU with a scalar slope.

## Math Interpretation

For each element `(i, j)` in the valid region:

$$ \mathrm{dst}_{i,j} = (\mathrm{src}_{i,j} > 0) ? \mathrm{src}_{i,j} : (\mathrm{src}_{i,j} \cdot \mathrm{slope}) $$

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
tlrelu %dst, %src, %slope : (!pto.tile<...>, !pto.tile<...>, f32)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TLRELU(TileData& dst, TileData& src0, typename TileData::DType scalar, WaitEvents&... events);
```

## Constraints

- The op iterates over `dst.GetValidRow()` / `dst.GetValidCol()`.

## Examples

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

void example() {
  using TileT = Tile<TileType::Vec, float, 16, 16>;
  TileT x, out;
  TLRELU(out, x, 0.1f);
}
```

