# 4. Tiles and GlobalTensor

This chapter defines the data-model contracts between tile operands and global memory operands. It specifies architecture-visible movement and interpretation rules.

## 4.1 Scope

This chapter covers:

- **Tile data model**: Structure and properties of tile values
- **GlobalTensor data model**: Global memory representation
- **GM-Tile movement**: TLOAD and TSTORE contracts
- **Shape compatibility**: Rules for movement and transform operations
- **Layout transforms**: Extract, insert, transpose, reshape

## 4.2 Tile Data Model

### 4.2.1 Tile Definition

A tile is the primary architectural data object for compute instructions. It represents a 2D array of elements stored in on-chip memory (Unified Buffer).

```
Tile Structure:
+-------------------+
|  d00  d01  d02 ...|  <- Row 0
|  d10  d11  d12 ...|
|  d20  d21  d22 ...|
|  ...  ...  ... ...|
+-------------------+
      Columns (C)
```

### 4.2.2 Tile Contract Components

A tile contract includes:

| Component | Description |
|-----------|-------------|
| Element type | Data type (f32, f16, i32, etc.) |
| Shape | Physical dimensions (RxC) |
| Valid rows (Rv) | Number of valid rows for computation |
| Valid cols (Cv) | Number of valid columns for computation |
| Location-intent | Tile class (Vec, Mat, Left, Right, Acc, Bias, Scale) |
| Layout | Row-major or column-major |

### 4.2.3 Tile Declaration

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Vector tile: general-purpose tile
using VecTile = Tile<TileType::Vec, float, 16, 16>;

// Matrix tile: for matrix operations
using MatTile = Tile<TileType::Mat, half, 16, 16>;

// Accumulator tile: for accumulation
using AccTile = Tile<TileType::Acc, float, 16, 16>;

// Bias tile: for bias addition
using BiasTile = Tile<TileType::Bias, float, 1, 16>;
```

### 4.2.4 Tile Type Parameters

From `include/pto/common/memory.hpp`:

```cpp
enum class TileType
{
    Vec,        // General-purpose vector tile
    Mat,        // Matrix operand
    Left,       // Left matrix for GEMM
    Right,      // Right matrix for GEMM
    Acc,        // Accumulator tile
    Bias,       // Bias tile
    Scaling,    // Scaling tile
    ScaleLeft,  // Left scale for MX operations
    ScaleRight, // Right scale for MX operations
};
```

## 4.3 GlobalTensor Data Model

### 4.3.1 GlobalTensor Definition

A GlobalTensor (or equivalent memory view) represents addressable global-memory data. It provides a view onto data stored in global memory (DRAM).

```
Global Memory (DRAM)
+------------------------------------------+
|  Tensor data (N-dimensional)             |
|  [batch, height, width, channels]        |
+------------------------------------------+
         |
         | TLOAD/TSTORE
         v
   Unified Buffer (UB)
+------------------------------------------+
|  Tile (2D view of global memory)         |
+------------------------------------------+
```

### 4.3.2 GlobalTensor Contract

Its architecture-visible contract includes:

| Component | Description |
|-----------|-------------|
| Element type | Must be compatible with tile operations |
| Shape | Tensor dimensions |
| Stride | Memory layout in each dimension |
| Address | Base address in global memory |
| Visibility | Memory ordering under synchronization |

### 4.3.3 GlobalTensor Declaration

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// 2D tensor in global memory
GlobalTensor<float> input{
    shape: {1024, 1024},    // M=1024, K=1024
    stride: {1024, 1},     // Row-major
    dtype: DataType::F32   // Float32
};

// 3D tensor (e.g., for convolution)
GlobalTensor<half> weight{
    shape: {256, 3, 3, 64},    // [out_channels, kernel_h, kernel_w, in_channels]
    stride: {576, 192, 64, 1},
    dtype: DataType::F16
};
```

### 4.3.4 Memory Reference Types

In IR/assembly, global memory is represented as:

```text
// 2D memory reference
!pto.memref<1024x1024xf32>

// 1D memory reference
!pto.memref<65536xi8>

// 3D memory reference with strides
!pto.memref<256x3x3x64xhalf>
```

## 4.4 GM-Tile Movement Contracts

### 4.4.1 TLOAD Operation

`TLOAD` transfers data from global memory to tile storage:

```
Global Memory                    Unified Buffer
+-------------+                  +-------------+
|  Data in   |   TLOAD          |             |
|  DRAM      | -------------->  |   Tile      |
|             |                  |  (Rv x Cv) |
+-------------+                  +-------------+
```

**Contract Requirements:**

```cpp
// TLOAD signature
template <typename TileData, typename GlobalTensor, typename... WaitEvents>
PTO_INST RecordEvent TLOAD(TileData& tile, const GlobalTensor& gmem, WaitEvents&... events);
```

Conforming implementations MUST preserve:

- Element mapping semantics in the defined valid domain
- Required ordering guarantees under event/TSYNC and memory model rules
- Documented behavior of quantization/scaling and mode attributes where present

### 4.4.2 TSTORE Operation

`TSTORE` transfers data from tile storage to global memory:

```
Unified Buffer                  Global Memory
+-------------+                  +-------------+
|             |   TSTORE         |             |
|   Tile      | -------------->  |  Data in    |
|  (Rv x Cv)  |                  |   DRAM      |
+-------------+                  +-------------+
```

**Contract Requirements:**

```cpp
// TSTORE signature
template <typename TileData, typename GlobalTensor, typename... WaitEvents>
PTO_INST RecordEvent TSTORE(const GlobalTensor& gmem, const TileData& tile, WaitEvents&... events);
```

### 4.4.3 Memory Operation Examples

```cpp
// Simple load
VecTile src;
TLOAD(src, gmem_input);

// Load with indices
VecTile dst;
TLOAD(dst, gmem_tensor[row_idx, col_idx]);

// Store
TSTORE(gmem_output, dst);

// Load with synchronization (Manual mode)
Event e;
TLOAD(src, gmem_input, e);
TSYNC(e);

// Store with atomic add
TSTORE(gmem_output, acc, AtomicType::AtomicAdd);
```

### 4.4.4 Quantized Operations

For vector quantization operations:

```cpp
// Store with scaling (vector quantization)
VecTile acc;      // Accumulator tile (f32)
VecTile scale;    // Scaling tile (f32/f16)
TSTORE_FP(gmem_output, acc, scale);
```

## 4.5 Shape and Domain Compatibility

### 4.5.1 Movement Compatibility Rules

For movement and layout-transform operations:

- Source and destination domains MUST satisfy instruction-specific compatibility constraints
- Out-of-domain behavior MUST be either explicitly defined (for example pad/fill) or declared unspecified
- Backend legality checks MUST reject unsupported shape/layout tuples deterministically

### 4.5.2 Compatibility Matrix

| Operation | Source Constraint | Destination Constraint |
|-----------|-----------------|----------------------|
| TLOAD | Global tensor shape | Tile shape (Rv, Cv) |
| TSTORE | Tile Rv x Cv | Global tensor shape |
| TEXTRACT | Source tile | Sub-tile shape |
| TINSERT | Sub-tile + offset | Destination tile |

### 4.5.3 Partial Tile Handling

When tile shape differs from operation requirements:

```cpp
// Example: Load 8x8 data into 16x16 tile
VecTile tile;
tile.SetValideRow(8);   // Only 8 rows valid
tile.SetValideCol(8);   // Only 8 columns valid

// TLOAD from 8x8 region
TLOAD(tile, gmem_small);  // Valid region = 8x8

// Operations only compute on valid region
TADD(dst, tile0, tile1);  // Computes 8x8 elements
```

### 4.5.4 Pad/Fill Operations

For explicit padding:

```cpp
// Fill padding region with zero
TFILLPAD(dst, src, PadValue::Zero);

// Fill with maximum value
TFILLPAD(dst, src, PadValue::Max);

// In-place fill
TFILLPAD_INPLACE(tile, PadValue::Zero);

// Expand: dst larger than src
TFILLPAD_EXPAND(dst, src, PadValue::Zero);
```

## 4.6 Layout Transform Operations

### 4.6.1 Transform Types

Operations such as extract/insert/reshape/transpose are architecture-level transforms over tile domains:

| Operation | Description | Index Mapping |
|-----------|-------------|---------------|
| TEXTRACT | Extract sub-tile | `(r, c) -> (r+roff, c+coff)` |
| TINSERT | Insert sub-tile | `(r, c) -> (r-roff, c-coff)` |
| TMOV | Copy with conversion | `(r, c) -> (r, c)` |
| TTRANS | Transpose | `(r, c) -> (c, r)` |
| TRESHAPE | Reinterpret shape | Same data, new dimensions |

### 4.6.2 Extract Operation

```cpp
// Extract 8x8 sub-tile from 16x16 tile at offset (4, 4)
VecTile src;      // 16x16 source
VecTile dst;      // 8x8 destination

TEXTRACT(dst, src, row_offset, col_offset);
```

### 4.6.3 Insert Operation

```cpp
// Insert 8x8 sub-tile into 16x16 tile at offset (4, 4)
VecTile src;      // 8x8 source
VecTile dst;      // 16x16 destination

TINSERT(dst, src, row_offset, col_offset);
```

### 4.6.4 Transpose Operation

```cpp
// Transpose 16x16 tile
VecTile src;
VecTile dst;

TTRANS(dst, src);
// After: dst[r][c] = src[c][r]
```

### 4.6.5 Reshape Operation

```cpp
// Reinterpret 16x16 tile as 8x32
VecTile src;
VecTile dst;

TRESHAPE(dst, src);
// Same data, different view
// Physical storage unchanged
```

### 4.6.6 Transform Contract Requirements

Layout transforms MUST define:

- Index-space mapping
- Valid-domain mapping
- Behavior for partially covered domains
- Implementation-defined constraints where hardware-specific behavior exists

## 4.7 Diagnostics Requirements

Movement/layout diagnostics SHOULD report:

### 4.7.1 Required Information

| Diagnostic Field | Description |
|-----------------|-------------|
| Offending operand | Which operand caused the error |
| Operation | The instruction that failed |
| Incompatible dimensions | Shape/layout/location mismatch |
| Index context | Relevant offset parameters |
| Error code | Deterministic identifier |

### 4.7.2 Example Diagnostics

```
Error [PTO-MEM-001]: Shape mismatch in TLOAD
  Operation: pto.tload
  Tile shape: 16x16
  Memory shape: 8x8
  Hint: Tile shape must match or be larger than loaded region

Error [PTO-MEM-002]: Invalid offset in TINSERT
  Operation: pto.tinsert
  Source shape: 8x8
  Dest shape: 16x16
  Offset: (12, 12)
  Hint: Offset + source shape exceeds destination bounds

Error [PTO-MEM-003]: Unsupported layout for TTRANS
  Operation: pto.ttrans
  Tile type: ColMajor
  Hint: Transpose requires row-major layout on A2/A3 backend
```

## 4.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Tile API | docs/coding/Tile.md |
| GlobalTensor API | docs/coding/GlobalTensor.md |
| TLOAD instruction | docs/isa/TLOAD.md |
| TSTORE instruction | docs/isa/TSTORE.md |
| Memory constants | include/pto/common/constants.hpp |
