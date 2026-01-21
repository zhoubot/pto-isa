# MSCATTER

## Introduction

Scatter-store elements from a tile into global memory using per-element indices.

## Math Interpretation

For each element `(i, j)` in the source valid region:

$$ \mathrm{mem}[\mathrm{idx}_{i,j}] = \mathrm{src}_{i,j} $$

If multiple elements map to the same destination location, the final value is implementation-defined (CPU simulator: last writer wins in row-major iteration order).

## Assembly Syntax

PTO-AS form: see `docs/grammar/PTO-AS.md`.

Synchronous form:

```text
mscatter %mem, %src, %idx : (!pto.tensor<...>, !pto.tile<...>, !pto.tile<...>)
```
## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename GlobalData, typename TileSrc, typename TileInd, typename... WaitEvents>
PTO_INST RecordEvent MSCATTER(GlobalData& dst, TileSrc& src, TileInd& indexes, WaitEvents&... events);
```

## Constraints

- Index interpretation is target-defined. The CPU simulator treats indices as linear element indices into `dst.data()`.
- No bounds checks are enforced on `indexes` by the CPU simulator.
