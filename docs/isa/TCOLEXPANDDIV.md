# TCOLEXPANDDIV

## Introduction

Column-wise broadcast divide.

This op divides `src0` by a per-column factor taken from the **first row** of `src1`.

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`.

For `0 <= i < R` and `0 <= j < C`:

$$ \mathrm{dst}_{i,j} = \frac{\mathrm{src0}_{i,j}}{\mathrm{src1}_{0,j}} $$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileDataDst, typename TileDataSrc1, typename... WaitEvents>
PTO_INST RecordEvent TCOLEXPANDDIV(TileDataDst& dst, TileDataDst& src0, TileDataSrc1& src1, WaitEvents&... events);
```

## Backend support

- CPU: **TODO**
- A2/A3: **TODO**
- A5: **Yes** (`include/pto/npu/a5/TColExpandDiv.hpp`)

## Constraints (from implementation)

### A5

- DType: `half` or `float`.
- Layout: `TileDataDst::isRowMajor`.
- Iteration domain: `dst.GetValidRow()` × `dst.GetValidCol()`.

### Shape / boundary conditions (semantic requirements)

For correctness (should be asserted by tests / runtime checks):
- `src0` valid shape equals `dst` valid shape.
- `src1` provides at least one row, and its first row has at least `dst.GetValidCol()` valid elements.

## Intrinsics (A5)

- Vector math: `vdiv`
- Loads/stores: `vlds`, `vsts` (inside the A5 helper)
