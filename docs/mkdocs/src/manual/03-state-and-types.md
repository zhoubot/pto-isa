# 3. State and Types

This chapter defines the architecture-visible state model and the type-level contracts that PTO Virtual ISA operations consume and produce.

## 3.1 Scope

This chapter covers:

- **Architectural state model**: The conceptual state visible to PTO programs
- **Type classes**: The type system used in PTO operations
- **Tile legality**: Constraints on tile operations
- **Valid-region semantics**: How partial tiles are handled
- **Attribute contracts**: Instruction modifiers and their semantics

## 3.2 Architectural State Model

The architecture models the following conceptual state:

### 3.2.1 Tile Values and Metadata

Tiles are the primary data containers in PTO:

```cpp
// Tile declaration example
using TileT = Tile<TileType::Vec, float, 16, 16>;
```

Tile metadata includes:
- **Element data**: The actual values stored in the tile
- **Valid rows (Rv)**: Number of valid rows for computation
- **Valid columns (Cv)**: Number of valid columns for computation
- **Shape (RxC)**: Physical tile dimensions

### 3.2.2 Scalar Values

Scalar values include:
- Immediate integers (i8, i16, i32, i64)
- Floating-point values (f16, f32, bf16)
- Index values (used for addressing)

```cpp
// Scalar types in PTO
int32_t scalar_i32;    // 32-bit signed integer
float scalar_f32;      // 32-bit float
uint32_t index_val;    // Index for memory addressing
```

### 3.2.3 Global Memory Views

Global memory views (GlobalTensor) represent addressable memory:

```cpp
// GlobalTensor declaration
GlobalTensor<float> gmem_input{shape: {1024, 1024}, stride: {1024, 1}};
```

### 3.2.4 Synchronization/Event State

Events are explicit dependency tokens:

```cpp
// Event usage
Event e;
TLOAD(tile, gmem, e);  // Load produces event e
TSYNC(e);              // Wait for event e
```

Backend-internal transient state is out of scope unless it changes architectural behavior.

## 3.3 Type Classes

PTO Virtual ISA type classes include:

### 3.3.1 Tile Types (`!pto.tile<...>`)

Tile types are parameterized by:

- **Element type (dtype)**: The data type of each element
- **Shape**: Rows x Columns (e.g., 16x16, 32x32)
- **Tile class**: Vec, Mat, Left, Right, Acc, Bias, Scale

```text
!pto.tile<16x16xf32>      // 16x16 float tile
!pto.tile<32x32xhalf>     // 32x32 half-precision tile
!pto.tile<16x16xi32>      // 16x16 32-bit integer tile
```

### 3.3.2 Memory/Reference Types (`!pto.memref<...>`)

Global memory references:

```text
!pto.memref<1024x1024xf32>    // 2D float memory
!pto.memref<65536xi8>         // 1D byte array
```

### 3.3.3 Scalar Types

MLIR builtin scalar types:

| Type | Description | Size |
|------|-------------|------|
| `i8`, `i16`, `i32`, `i64` | Signed integers | 8-64 bits |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers | 8-64 bits |
| `f16`, `f32`, `bf16` | Floating-point | 16-32 bits |
| `index` | Platform-specific index | Implementation-defined |

### 3.3.4 Event Types (`!pto.event`)

Event types for synchronization:

```text
!pto.event    // Synchronization token
```

Each instruction family MUST define accepted type classes for each operand/result position.

## 3.4 Tile Legality Dimensions

Tile legality is constrained by multiple dimensions:

### 3.4.1 Element Type (dtype)

Supported element types vary by backend profile:

| dtype | Description | A2/A3 Support | A5 Support |
|-------|-------------|---------------|------------|
| `f32` | 32-bit float | Yes | Yes |
| `f16` | 16-bit float | Yes | Yes |
| `bf16` | Brain float | Yes | Yes |
| `i32` | 32-bit integer | Yes | Yes |
| `i16` | 16-bit integer | Yes | Yes |
| `i8` | 8-bit integer | Partial | Yes |
| `u32` | 32-bit unsigned | Yes | Yes |
| `u16` | 16-bit unsigned | Yes | Yes |
| `u8` | 8-bit unsigned | Partial | Yes |

### 3.4.2 Shape and Valid-Region Compatibility

Shapes must satisfy instruction-specific constraints:

```cpp
// Example: Valid shape combinations
Tile<TileType::Vec, float, 16, 16> tile_v;   // 16x16 vector tile
Tile<TileType::Mat, half, 16, 16> tile_m;     // 16x16 matrix tile
Tile<TileType::Acc, float, 16, 16> tile_acc;  // 16x16 accumulator
```

### 3.4.3 Location-Intent Role

Tile class (TileType) participates in instruction legality:

| TileType | Description | Legal Operations |
|----------|-------------|------------------|
| `Vec` | General-purpose vector | Elementwise, reduce, memory |
| `Mat` | Matrix operand | Matrix multiply, elementwise |
| `Left` | Left matrix | TMATMUL |
| `Right` | Right matrix | TMATMUL |
| `Acc` | Accumulator | TMATMUL, elementwise |
| `Bias` | Bias tile | TGEMV_BIAS, TMATMUL_BIAS |
| `Scale` | Scaling tile | Quantized operations |
| `ScaleLeft` | Left scale | MX operations |
| `ScaleRight` | Right scale | MX operations |

### 3.4.4 Layout Class

Tile layout affects memory access patterns:

| Layout | Description |
|--------|-------------|
| `RowMajor` | Elements contiguous in row direction |
| `ColMajor` | Elements contiguous in column direction |

The virtual ISA defines the legality interface; concrete support sets are backend-profile-specific.

## 3.5 Valid-Region Semantics

Valid-region semantics are first-class in PTO:

### 3.5.1 Definition

- Semantic definitions apply to indices in the declared valid domain (0 <= r < Rv, 0 <= c < Cv)
- Values outside valid domain are unspecified unless explicitly defined
- Multi-operand operations MUST define domain compatibility rules

### 3.5.2 Standard Notation

The standard notation uses `Rv` and `Cv` for valid rows/columns:

```
Tile with Rv=8, Cv=16 (physical 16x16):
+---------------------------+
| Rv x Cv = 8x16 elements  |  <- Valid region (computed)
|                          |
|                          |
|                          |
+---------------------------+
|    Unspecified region    |  <- Outside valid domain
|    (may contain any     |
|     value or be unused)  |
+---------------------------+
```

### 3.5.3 Valid Region Operations

```cpp
// Setting valid region
TileT tile;
// Set valid region: 8 rows, 16 columns valid
tile.SetValideRow(8);
tile.SetValideCol(16);

// Valid region is used for:
// - Compute operations (results only valid in this region)
// - Memory movement (only valid region is meaningful)
// - Reduction operations (reduce over valid region)
```

### 3.5.4 Compatibility Rules

When multiple tiles are used in an operation:

| Operation Type | Compatibility Rule |
|---------------|-------------------|
| Elementwise | All operands must have same Rv and Cv |
| Reduction | Output Rv/Cv determined by reduction axis |
| Memory load | Tile Rv/Cv set from memory shape |

## 3.6 Attribute Contracts

Instruction attributes modify operation behavior:

### 3.6.1 Compare Mode (`cmpMode`)

Used in comparison operations:

| Value | Description |
|-------|-------------|
| `EQ` | Equal |
| `NE` | Not equal |
| `LT` | Less than |
| `LE` | Less than or equal |
| `GT` | Greater than |
| `GE` | Greater than or equal |

Example:
```text
%result = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

### 3.6.2 Rounding Mode (`rmode`)

Used in type conversion:

| Value | Description |
|-------|-------------|
| `CAST_NONE` | No rounding |
| `CAST_RINT` | Round to nearest, ties to even |
| `CAST_ROUND` | Round to nearest, ties away from zero |
| `CAST_FLOOR` | Round toward negative infinity |
| `CAST_CEIL` | Round toward positive infinity |
| `CAST_TRUNC` | Round toward zero |
| `CAST_ODD` | Von Neumann rounding (round to odd) |

### 3.6.3 Mask Pattern (`maskMode`)

Used in reduction operations:

| Value | Description |
|-------|-------------|
| `P0101` | Take every 2nd element (start with 1st) |
| `P1010` | Take every 2nd element (start with 2nd) |
| `P0001` | Take every 4th element (start with 1st) |
| `P0010` | Take every 4th element (start with 2nd) |
| `P0100` | Take every 4th element (start with 3rd) |
| `P1000` | Take every 4th element (start with 4th) |
| `P1111` | Take all elements |

### 3.6.4 Attribute Requirements

Each attribute MUST define:
- Type/domain constraints
- Default behavior (if any)
- Interaction with semantics and legality checks
- Diagnostics requirements for invalid values

## 3.7 Diagnostics Requirements

Type/state verification diagnostics SHOULD include:

### 3.7.1 Required Information

- Operand position (which argument failed)
- Expected type class and received type class
- Relevant legality dimensions (dtype/layout/location/shape)
- Deterministic error identifiers for CI stability

### 3.7.2 Example Diagnostics

```
Error [PTO-VAL-001]: Invalid tile shape for TMATMUL
  Operation: pto.tmatmul
  Expected: square tile (rows == cols)
  Received: 16x32
  Hint: See docs/coding/debug.md and search for TMATMUL constraints

Error [PTO-VAL-002]: Unsupported dtype for TADD
  Operation: pto.tadd
  Expected: f32, f16, i32, i16
  Received: i64
  Hint: See docs/coding/debug.md and search for TADD dtype constraints
```

## 3.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Tile API | docs/coding/Tile.md |
| Scalar Types | docs/coding/Scalar.md |
| Event API | docs/coding/Event.md |
| Instruction constraints | docs/isa/*.md |
| Debugging | docs/coding/debug.md |
