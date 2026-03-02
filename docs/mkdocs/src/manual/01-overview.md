# 1. Overview

This chapter provides an introduction to the PTO (Parallel Tile Operation) Virtual Instruction Set Architecture, including its design goals, architectural identity, and scope.

## 1.1 Design Goals

The PTO Virtual ISA is designed with the following primary objectives:

### 1.1.1 Architectural Stability

PTO provides a stable architecture contract that abstracts hardware differences across evolving Ascend generations (A2, A3, A5). This allows:

- Software written for one generation to port to newer generations with minimal changes
- Hardware evolution without breaking user software contracts
- Clear separation between architecture-defined and implementation-defined behavior

### 1.1.2 Tile-Centric Semantics

The tile is the fundamental unit of computation in PTO. All instruction semantics are defined over tile domains, which provides:

- Predictable behavior across different tile sizes (e.g., 16x16, 32x32)
- Explicit valid-region handling through `Rv` (valid rows) and `Cv` (valid columns)
- Clear semantics for partial tile operations

### 1.1.3 Practical Bridge

PTO serves as a practical bridge from high-level programming to backend code generation:

- Direct mapping to hardware instructions
- Support for both compiler-managed (Auto) and programmer-managed (Manual) modes
- Integration with MLIR-based toolchains (PTO-AS, PTO IR, bytecode)

## 1.2 PTO Architectural Identity

PTO distinguishes itself from generic GPU ISAs by making the following architecture concepts first-class:

### 1.2.1 Tile as Primary Compute Unit

All compute instructions operate on tile domains. A tile represents a 2D array of elements stored in on-chip memory (SRAM).

```
Tile Structure:
+---------------------------+
| Rv x Cv elements          |  <- Valid region (computed)
| (shape: RxC)              |
+---------------------------+
```

Example tile declaration:
```cpp
using TileT = Tile<TileType::Vec, float, 16, 16>;  // 16x16 float tile
```

### 1.2.2 Valid-Region-First Semantics

PTO mandates that all tile operations define behavior within the valid region:

- **Valid Rows (Rv)**: Number of rows with valid data
- **Valid Columns (Cv)**: Number of columns with valid data
- Operations outside the valid region have unspecified behavior unless explicitly defined

This is critical for handling variable-sized computations in neural networks.

### 1.2.3 Location-Intent Model

Tiles have location-intent roles that participate in instruction legality rules:

| Tile Class | Description | Typical Usage |
|------------|-------------|---------------|
| `Mat` | Matrix operand | GEMM left/right matrices |
| `Left` | Left matrix | Matrix multiply operand |
| `Right` | Right matrix | Matrix multiply operand |
| `Acc` | Accumulator | Accumulation tile for GEMM |
| `Bias` | Bias tile | Bias addition |
| `Scale` | Scale tile | Quantization scaling |
| `Vec` | Vector tile | General-purpose tile |

### 1.2.4 Dual Programming Model

PTO supports two complementary programming models:

**Auto Mode**: The compiler/runtime manages:
- Tile placement on physical storage
- Synchronization insertion
- Operation scheduling

**Manual Mode**: The programmer controls:
- Explicit tile addressing via `TASSIGN`
- Event-based synchronization via `TSYNC`
- Operation ordering

Both modes are architecturally valid and interoperable.

### 1.2.5 Event-Centric Synchronization

PTO uses an explicit event model for dependency management:

```cpp
// Manual mode: explicit synchronization
Event e;
TLOAD(tile0, gmem0, e);    // Load with event
TLOAD(tile1, gmem1, e);    // Same event = same dependency group
TSYNC(e);                   // Wait for all operations with event e
TMATMUL(acc, tile0, tile1); // Compute after synchronization
```

## 1.3 Architecture Boundary

### 1.3.1 Architecture-Defined Behavior

The PTO architecture explicitly defines:

- **Instruction semantics**: Observable results in valid regions
- **Ordering semantics**: Required synchronization and happens-before relations
- **Type system**: Legal type combinations and constraints
- **Legal boundaries**: What behaviors are guaranteed and observable

### 1.3.2 Implementation-Defined Behavior

These areas are implementation-defined and vary by backend:

- Microarchitectural scheduling details
- Exact on-chip storage layout
- Cache behavior and buffering strategies
- Performance characteristics

> **Note**: Backend-specific details MUST be documented as implementation-defined constraints in each backend profile (see Chapter 12).

### 1.3.3 Undefined Behavior

The following are explicitly undefined:

- Operations on indices outside valid region (unless specified)
- Use of uninitialized tiles
- Race conditions without proper synchronization
- Access to unmapped memory

## 1.4 Source of Truth

Authoritative PTO sources are maintained in specific locations:

### 1.4.1 Instruction Semantics

Per-instruction specifications are in `docs/isa/*.md`:

```
docs/isa/
  TADD.md      # Elementwise addition
  TMATMUL.md   # Matrix multiply
  TLOAD.md     # Load from global memory
  ...          # Other instructions
```

Each instruction document includes:
- Mathematical interpretation
- Assembly syntax
- C++ intrinsic signature
- Constraints and valid region behavior

### 1.4.2 Public API

C++ intrinsics are defined in `include/pto/common/pto_instr.hpp`:

```cpp
// Example: TADD intrinsic declaration
template <typename TileData, typename... WaitEvents>
PTO_INST RecordEvent TADD(TileData& dst, TileData& src0, TileData& src1, WaitEvents&... events);
```

### 1.4.3 Assembly Grammar

The PTO-AS textual format is defined in:
- `docs/grammar/PTO-AS.md` - Human-readable specification
- `docs/grammar/PTO-AS.bnf` - Formal BNF grammar

## 1.5 Instruction-Family Taxonomy

PTO instructions are organized into families based on their functionality:

### 1.5.1 Synchronization and Resource Binding

| Instruction | Description |
|-------------|-------------|
| `TSYNC` | Synchronization barrier |
| `TASSIGN` | Tile-to-address frontend |
| `TSETHF32MODE` | HF32 mode configuration |
| `TSETTF32MODE` | TF32 mode configuration |

### 1.5.2 Elementwise Operations

Binary operations on tile operands:
- Arithmetic: `TADD`, `TSUB`, `TMUL`, `TDIV`
- Bitwise: `TAND`, `TOR`, `TXOR`, `TSHL`, `TSHR`
- Comparison: `TCMP`, `TMIN`, `TMAX`
- Mathematical: `TLOG`, `TEXP`, `TSQRT`, `TRSqrt`, `TRECIP`
- Activation: `TRELU`, `TPRELU`, `TLRELU`

### 1.5.3 Tile-Scalar Operations

Operations between tiles and scalar values:
- `TADDS`, `TSUBS`, `TMULS`, `TDIVS`
- `TEXPANDS` (broadcast scalar to tile)
- `TCMPS`, `TSELS`

### 1.5.4 Reduction Operations

Axis-based reduction:
- Row: `TROWSUM`, `TROWPROD`, `TROWMAX`, `TROWMIN`
- Column: `TCOLSUM`, `TCOLPROD`, `TCOLMAX`, `TCOLMIN`

### 1.5.5 Memory Operations

Global memory ↔ Tile data movement:
- `TLOAD`: Load from global memory to tile
- `TSTORE`: Store from tile to global memory
- `MGATHER` / `MSCATTER`: Indexed access

### 1.5.6 Matrix Multiply

GEMM operations:
- `TMATMUL`: Basic matrix multiply
- `TMATMUL_ACC`: Fused multiply-accumulate
- `TMATMUL_BIAS`: With bias addition
- `TMATMUL_MX`: Mixed-precision/quantized

### 1.5.7 Data Movement and Layout

Tile transformation operations:
- `TEXTRACT` / `TINSERT`: Sub-tile extraction and insertion
- `TMOV`: Tile copy with optional conversion
- `TTRANS`: Tile transpose
- `TRESHAPE`: Tile shape reinterpretation

### 1.5.8 Complex Operations

Specialized operations:
- `TPRINT`: Debug output
- `TQUANT`: Quantization
- `TGATHER` / `TSCATTER`: Element gather/scatter

Family-level contracts are defined in Chapter 7 (Instruction Set). Per-op semantics remain in `docs/isa/*.md`.

## 1.6 Compatibility Principles

### 1.6.1 Additive Evolution

New versions SHOULD prefer additive changes:

- Adding new instructions
- Adding new optional attributes
- Extending supported type sets

### 1.6.2 Breaking Changes

When breaking changes are necessary:

- MUST include explicit versioning (e.g., v1, v2)
- MUST provide migration guidance
- MUST document transition path

### 1.6.3 Implementation-Defined Behavior

All implementation-defined behavior MUST be:

- Explicitly tagged in documentation
- Consistent across IR, assembly, and backend layers
- Documented per backend profile

## 1.7 Example Programs

### 1.7.1 Auto Mode Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Compiler manages placement and synchronization
void kernel_add() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    
    // Load data (compiler inserts synchronization)
    TLOAD(src0, gmem_input0);
    TLOAD(src1, gmem_input1);
    
    // Compute
    TADD(dst, src0, src1);
    
    // Store result (compiler inserts synchronization)
    TSTORE(gmem_output, dst);
}
```

### 1.7.2 Manual Mode Example

```cpp
#include <pto/pto-inst.hpp>

using namespace pto;

// Programmer controls placement and synchronization
void kernel_add_manual() {
    using TileT = Tile<TileType::Vec, float, 16, 16>;
    TileT src0, src1, dst;
    
    // Explicit tile placement
    TASSIGN(src0, 0x1000);  // Address in UB
    TASSIGN(src1, 0x2000);
    TASSIGN(dst,  0x3000);
    
    // Explicit event-based synchronization
    Event e0, e1;
    TLOAD(src0, gmem_input0, e0);
    TLOAD(src1, gmem_input1, e1);
    
    TSYNC(e0);
    TSYNC(e1);
    
    // Compute
    TADD(dst, src0, src1);
    
    // Store with synchronization
    Event e2;
    TSTORE(gmem_output, dst, e2);
    TSYNC(e2);
}
```

## 1.8 Related Documentation

| Chapter | Description |
|---------|-------------|
| Chapter 2 | Abstract machine model |
| Chapter 3 | State and type system |
| Chapter 4 | Tiles and GlobalTensor |
| Chapter 5 | Synchronization |
| Chapter 6 | PTO Assembly (PTO-AS) |
| Chapter 7 | Instruction set overview |
| Chapter 9 | Virtual ISA and IR |
| Chapter 10 | Bytecode and toolchain |
