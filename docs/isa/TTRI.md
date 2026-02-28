# TTRI

## Introduction

Fill a triangular 0/1 mask into `dst` (upper or lower), with an optional diagonal offset.

`TTRI` produces a mask tile:

- lower-triangular (`isUpperOrLower=0`): ones on and below the diagonal
- upper-triangular (`isUpperOrLower=1`): ones on and above the diagonal

## Math Interpretation

Let `R = dst.GetValidRow()` and `C = dst.GetValidCol()`.

Lower-triangular:

$$
\mathrm{dst}_{i,j} = \begin{cases}
1 & j \le i + \mathrm{diagonal} \\
0 & \text{otherwise}
\end{cases}
$$

Upper-triangular:

$$
\mathrm{dst}_{i,j} = \begin{cases}
1 & j \ge i + \mathrm{diagonal} \\
0 & \text{otherwise}
\end{cases}
$$

## C++ Intrinsic

Declared in `include/pto/common/pto_instr.hpp`:

```cpp
template <typename TileData, int isUpperOrLower, int diagonal, typename... WaitEvents>
PTO_INST RecordEvent TTRI(TileData& dst, WaitEvents&... events);
```

## Backend support

- CPU: **TODO**
- A2/A3: **Yes** (`include/pto/npu/a2a3/TTri.hpp`)
- A5: **TODO**

## Constraints (from implementation)

### A2/A3

- DType: one of `int32_t/int16_t/uint32_t/uint16_t/half/float`.
- Layout: row-major.
- Template parameter: `isUpperOrLower` must be 0 or 1.

## Intrinsics (A2/A3)

- mask controls: `set_vector_mask`, `set_mask_count`, `set_mask_norm`
- fill: `vector_dup`
- ordering: `pipe_barrier(PIPE_V)`
