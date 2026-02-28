# 6. PTO Assembly (PTO-AS)

This chapter defines the Virtual ISA contract of PTO-AS as the textual form of PTO programs.

## 6.1 Scope

This chapter covers:

- **Core form**: Instruction structure and syntax
- **Operand classes**: Tile, memory, scalar, event operands
- **Attribute system**: Instruction modifiers
- **Directives**: Program structure elements
- **Structural validity**: Grammar rules
- **Diagnostics**: Error reporting

The normative grammar remains in:
- `docs/grammar/PTO-AS.md` (human-readable specification)
- `docs/grammar/PTO-AS.bnf` (formal BNF grammar)

## 6.2 Core Form

### 6.2.1 Instruction Structure

PTO-AS uses an instruction-centric SSA-like textual form. A typical statement shape is:

```text
%dst = tadd %src0, %src1 : (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32>;
```

**Components:**

| Component | Description |
|-----------|-------------|
| `%dst` | Result SSA value (optional) |
| `tadd` | Operation opcode |
| `%src0, %src1` | Input SSA values |
| `{...}` | Attribute dictionary (optional) |
| `: ... -> ...` | Type signature (optional but recommended) |

### 6.2.2 SSA Value Naming

PTO-AS uses SSA-like value names following MLIR conventions:

```text
// Single result
%result = tadd %a, %b : ...;

// Multiple results (if supported)
%result0, %result1 = tload %mem : ... -> ...;
```

### 6.2.3 Synchronous Execution Model

PTO-AS is a synchronous, line-ordered format:

- No `wait(...)` clause
- No implicit event result
- Explicit dependencies use explicit instructions (e.g., `tsync`)

```text
// Explicit synchronization
%e0 = tload %mem, %arg1 : ... -> ...;    // Load produces event
tsync %e0;                               // Wait for load
%r = tadd %a, %b : ...;                  // Compute after sync
```

### 6.2.4 Type Signatures

Type signatures are recommended for readability but may be omitted when types are unambiguous:

```text
// With full type signature
%dst = tadd %src0, %src1 : (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32>;

// With abbreviated type signature
%dst = tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>;

// Without type signature (types must be inferred)
%dst = tadd %src0, %src1;
```

## 6.3 Operand Classes

### 6.3.1 Tile Operands

Tile operands are SSA values of tile type:

```text
%tile = tadd %src0, %src1 : (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32>;
```

### 6.3.2 Memory Operands

Memory operands reference global memory:

```text
// Simple memory operand
%tile = tload %mem : (!pto.memref<1024x1024xf32>) -> !pto.tile<16x16xf32>;

// Indexed memory operand (with row/column offsets)
%tile = tload %mem[%row_idx, %col_idx] : (!pto.memref<...>, index, index) -> !pto.tile<...>;
```

Memory operands support indexed access similar to PTX:

```text
%t0 = tload %gmem[%c0, %c1] : (!pto.memref<1024x1024xf32>, index, index) -> !pto.tile<16x16xf32>;
```

### 6.3.3 Scalar Operands

Scalar operands include immediate values and constants:

```text
// Tile-scalar operation
%dst = tadds %tile, %scalar : (!pto.tile<16x16xf32>, f32) -> !pto.tile<16x16xf32>;

// With integer immediate
%dst = tshl %a, %b, 4 : ...;
```

### 6.3.4 Event Operands

Event operands express dependencies:

```text
// Load with event
%e0 = tload %mem : (!pto.memref<...>) -> !pto.tile<...>;

// Wait for event
tsync %e0;

// Multiple events
tsync %e0;
tsync %e1;
```

### 6.3.5 Operand Summary Table

| Operand Class | Syntax | Example |
|---------------|--------|---------|
| Tile | SSA value | `%tile` |
| Memory | SSA value | `%mem` |
| Memory (indexed) | `%mem[%idx0, %idx1]` | `%gmem[%r, %c]` |
| Scalar | SSA or literal | `%scalar`, `4`, `3.14` |
| Event | SSA value | `%e0` |

## 6.4 Attribute and Modifier Contract

### 6.4.1 Attribute Dictionary

Instruction modifiers not expressed as positional operands use MLIR-style attribute dictionaries:

```text
%result = tcmp %a, %b {cmpMode = #pto.cmp<GT>} : !pto.tile<16x16xf32> -> !pto.tile<16x16xi1>;
```

### 6.4.2 Compare Mode Attribute

```text
{cmpMode = #pto.cmp<EQ>}   // Equal
{cmpMode = #pto.cmp<NE>}   // Not equal
{cmpMode = #pto.cmp<LT>}   // Less than
{cmpMode = #pto.cmp<LE>}   // Less than or equal
{cmpMode = #pto.cmp<GT>}   // Greater than
{cmpMode = #pto.cmp<GE>}   // Greater than or equal
```

### 6.4.3 Rounding Mode Attribute

```text
{rmode = #pto.round<CAST_NONE>}   // No rounding
{rmode = #pto.round<CAST_RINT>}   // Round to nearest, ties to even
{rmode = #pto.round<CAST_ROUND>}  // Round to nearest, ties away from zero
{rmode = #pto.round<CAST_FLOOR>}  // Round toward negative infinity
{rmode = #pto.round<CAST_CEIL>}   // Round toward positive infinity
{rmode = #pto.round<CAST_TRUNC>}  // Round toward zero
```

### 6.4.4 Mask Pattern Attribute

```text
{maskMode = #pto.mask<P1111>}   // Take all elements
{maskMode = #pto.mask<P0101>}   // Take every 2nd element
{maskMode = #pto.mask<P1010>}   // Take every 2nd element (offset)
```

### 6.4.5 Attribute Requirements

Each attribute MUST define:

| Requirement | Description |
|-------------|-------------|
| Name | Attribute identifier |
| Type | Value domain (enum, integer, float, etc.) |
| Default | Default value policy (if any) |
| Semantic impact | How it affects instruction behavior |
| Diagnostics | Behavior for invalid values |

## 6.5 Directives

### 6.5.1 Argument Declaration

Declares external inputs (function arguments):

```text
.arg %a : !pto.tile<16x16xf16>;
.arg %b : !pto.tile<16x16xf16>;
.arg %c : !pto.event;
```

### 6.5.2 Event Arguments

Event arguments for explicit dependencies:

```text
.arg %e0 : !pto.event;
.arg %e1 : !pto.event;
```

### 6.5.3 Constant Declaration

Introduces SSA values with constant values:

```text
.const %c0 = 0 : index;
.const %c1 = 1 : index;
.const %stride = 1024 : index;
.const %alpha = 1.0 : f32;
.const %beta = 0.0 : f32;
```

### 6.5.4 Complete Program Example

```text
// Function arguments
.arg %lhs : !pto.tile<16x16xf16>;
.arg %rhs : !pto.tile<16x16xf16>;
.arg %acc : !pto.tile<16x16xf32>;

// Constants
.const %c0 = 0 : index;
.const %c1 = 1 : index;

// Load with events
%e0 = tload %lhs : (!pto.memref<...>) -> !pto.tile<...>;
%e1 = tload %rhs : (!pto.memref<...>) -> !pto.tile<...>;

// Synchronize
tsync %e0;
tsync %e1;

// Compute matrix multiply
%result = tmatmul %acc, %lhs, %rhs : ...;

// Store
%e2 = tstore %mem, %result : ...;

// Final sync
tsync %e2;
```

## 6.6 Structural Validity Rules

### 6.6.1 Required Conditions

A structurally valid PTO-AS program MUST satisfy:

1. **Operand/Result arity consistency**: Number of operands matches instruction signature
2. **Type-class compatibility**: Operand types match operation contract
3. **Required attribute presence**: All required attributes are provided
4. **Parseable forms**: Statement forms match grammar rules

### 6.6.2 Validity Rules by Category

| Category | Rule |
|----------|------|
| Arity | Operand count must match instruction definition |
| Types | Operands must be of compatible type class |
| Attributes | Required attributes must be present |
| SSA | All referenced values must be defined |
| Events | Events must be properly produced and consumed |

### 6.6.3 Example Validity Errors

```text
// Error: Wrong operand count
%dst = tadd %a : (!pto.tile<...>) -> !pto.tile<...>;
// Expected 2 operands, got 1

// Error: Type mismatch
%dst = tadd %a, %b : (!pto.tile<16x16xf32>, !pto.tile<16x16xi32>) -> ...;
// Operand types must match

// Error: Missing required attribute
%dst = tcmp %a, %b : ...;
// Compare mode attribute required

// Error: Undefined SSA value
%dst = tadd %undefined, %b : ...;
// %undefined not defined in program
```

## 6.7 Diagnostics Contract

### 6.7.1 Requirements

PTO-AS diagnostics MUST be:

- **Location-aware**: Include file, line, and column for parse errors
- **Deterministic**: Same input produces same error message
- **Actionable**: Include expected vs. actual information

### 6.7.2 Error Classes

| Error Class | Description |
|-------------|-------------|
| `PTO-PARSE-001` | Syntax error |
| `PTO-PARSE-002` | Invalid token |
| `PTO-VAL-001` | Type mismatch |
| `PTO-VAL-002` | Arity mismatch |
| `PTO-VAL-003` | Missing attribute |
| `PTO-SSA-001` | Undefined value reference |

### 6.7.3 Example Diagnostics

```
Error [PTO-PARSE-001] at line 5: Unexpected token
  Expected: ',' or ')'
  Received: ';'
  Context: %dst = tadd %a %b;

Error [PTO-VAL-002] at line 10: Operand arity mismatch
  Operation: tadd
  Expected: 2 operands
  Received: 1 operand

Error [PTO-VAL-003] at line 15: Missing required attribute
  Operation: tcmp
  Required attribute: cmpMode
```

## 6.8 Compatibility and Evolution

### 6.8.1 Evolution Policy

PTO-AS evolution SHOULD be additive:

- Adding new instructions
- Adding new optional attributes
- Extending supported type sets

### 6.8.2 Breaking Changes

Breaking textual-syntax changes MUST be:

- Versioned (e.g., PTO-AS v1, v2)
- Accompanied by migration guidance
- Rejected with deterministic diagnostics in older versions

### 6.8.3 Rejection Policy

Toolchains MUST reject unsupported syntax with deterministic diagnostics:

```text
Error [PTO-VERSION-001]: Unsupported PTO-AS version
  Expected: PTO-AS v1.x or v2.x
  Received: PTO-AS v0.x
  Hint: See migration guide for v0.x to v1.0
```

## 6.9 IR Type System

### 6.9.1 Type Representations

PTO-AS uses MLIR-like type spellings:

| Type | IR Notation | Description |
|------|-------------|-------------|
| Tile | `!pto.tile<RxCxdtype>` | 2D tile |
| Memory | `!pto.memref<...xdtype>` | Global memory reference |
| Event | `!pto.event` | Synchronization token |
| Integer | `i8`, `i16`, `i32`, `i64` | Signed integer |
| Float | `f16`, `f32` | Floating-point |
| Index | `index` | Platform-specific index |

### 6.9.2 Type Examples

```text
// 16x16 float32 tile
!pto.tile<16x16xf32>

// 32x32 float16 tile
!pto.tile<32x32xf16>

// 2D memory reference
!pto.memref<1024x1024xf32>

// 1D memory reference
!pto.memref<65536xi8>

// Scalar types
f32, i32, index
```

## 6.10 Related Documentation

| Topic | Reference |
|-------|-----------|
| Grammar (BNF) | docs/grammar/PTO-AS.bnf |
| Grammar (Human-readable) | docs/grammar/PTO-AS.md |
| Virtual ISA and IR | Chapter 9 |
| Bytecode and Toolchain | Chapter 10 |
| Instruction Reference | docs/isa/*.md |
