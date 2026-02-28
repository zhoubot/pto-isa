# 9. Virtual ISA and IR

This chapter defines the contract between PTO Virtual ISA semantics and PTO IR/lowering pipelines. The terms `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative as defined in RFC 2119.

## 9.1 Scope

This chapter covers:

- **Layering model**: Three-layer architecture (Virtual ISA, IR, Backend)
- **IR object model**: Structured representation requirements
- **Verifier boundary**: Split between structural and target verification
- **Lowering invariants**: Semantic preservation requirements
- **Source alignment**: Documentation synchronization
- **Compatibility policy**: IR evolution rules

## 9.2 Layering Model

PTO uses a three-layer contract to separate concerns:

### 9.2.1 Layer Architecture

```
+-------------------+
|   User Code       |  <- PTO-DSL, PyPTO, TileLang
+-------------------+
         |
         v
+-------------------+
|  Virtual ISA      |  <- PTO-AS, PTO IR
+-------------------+
         |
         v
+-------------------+
|   IR Layer        |  <- MLIR-based representation
+-------------------+
         |
         v
+-------------------+
| Backend Lowering  |  <- Ascend A2/A3/A5, CPU
+-------------------+
```

### 9.2.2 Layer Responsibilities

| Layer | Responsibility | Key Artifacts |
|-------|---------------|---------------|
| **Virtual ISA** | Architecture-visible semantics | Instruction contracts, memory model |
| **IR Layer** | Structured typed representation | Operation definitions, verification |
| **Backend** | Target-specific legalization | Code generation, optimization |

### 9.2.3 Layer Contract

Backend specialization MUST preserve Virtual ISA-observable behavior:

- Valid-region semantics must be preserved
- Explicit ordering dependencies must be maintained
- Operation meaning within architecture-defined domains must be preserved

```text
// Example: Layer preservation
Virtual ISA:  %r = tadd %a, %b : (tile<f32>, tile<f32>) -> tile<f32>
// IR:         "tadd"(%a, %b) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
// Backend:    Vector instruction (if supported) or scalar loop
// Constraint: All layers must produce same computational result
```

## 9.3 IR Object Model

### 9.3.1 Required Components

A conforming PTO IR model SHOULD define:

| Component | Description | Requirements |
|-----------|-------------|--------------|
| Module | Top-level container | Contains functions and symbols |
| Function | Callable unit | Has blocks, arguments, return types |
| Block | Basic unit of control flow | Has operations, terminator |
| Operation | Computational unit | Has name, operands, results, regions |
| SSA Value | Named computation result | Has type, definition point |

### 9.3.2 Module Contract

```mlir
module @module_name {
  func @function_name(%arg0: !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32> {
    %result = "tadd"(%arg0, %arg0) : (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32>
    return %result : !pto.tile<16x16xf32>
  }
}
```

### 9.3.3 Operation Schema

Every PTO operation MUST define:

```text
Operation Schema:
  - Name: Operation identifier (e.g., "tadd")
  - Operands: Input SSA values with types
  - Results: Output SSA values with types
  - Attributes: Optional modifiers (cmpMode, rmode, etc.)
  - Effects: Memory, synchronization side effects
```

### 9.3.4 Synchronization Effects

PTO operations MUST explicitly declare synchronization effects:

```mlir
// Load produces event (memory read + sync token)
%event = "tload"(%mem) : (!pto.memref<...>) -> !pto.tile<...>

// Sync consumes/produces ordering
"tsync"(%event) : (!pto.event) -> ()

// Store consumes tile + event, produces event
%result = "tstore"(%mem, %tile, %event) : ...
```

## 9.4 Verifier Boundary

Verification is split into two levels to ensure correctness at each layer:

### 9.4.1 Structural Verifier (IR Level)

The structural verifier validates target-independent properties:

**Requirements:**

- MUST validate operation schema (name, operands, results)
- MUST validate arity (operand/result counts match definition)
- MUST validate type classes (operand types are valid)
- MUST validate required attributes (all required attrs present)
- MUST be target-independent (no backend-specific rules)

**Example:**

```text
// Structural verification
Operation: tadd
Expected operands: 2 tile types
Received: 1 operand

Error [PTO-STR-001]: Operand count mismatch
  Operation: tadd
  Expected: 2 operands
  Received: 1 operand
```

### 9.4.2 Target Legality Verifier (Backend Level)

The target verifier validates backend-specific properties:

**Requirements:**

- MUST validate dtype/layout/location/shape tuples for selected backend profile
- MUST produce deterministic diagnostics for unsupported tuples
- MUST check profile-specific constraints

**Example:**

```text
// Target legality verification
Backend: Ascend A2
Operation: tmatmul
Operand: tile<32x32xf32>

Error [PTO-LEGAL-001]: Unsupported tile shape for A2
  Operation: tmatmul
  Backend: A2
  Supported shapes: 8x8, 16x16
  Received: 32x32
```

### 9.4.3 Verification Pipeline

```
Source Code (PTO-DSL/PyPTO)
        |
        v
   PTO-AS Text
        |
        v
 PTO IR (MLIR)
        |
        v
Structural Verifier  <-- Target-independent
        |
        v
Target-Specific IR  <-- Backend lowering
        |
        v
Target Legality Verifier  <-- Profile-specific
        |
        v
Generated Code
```

## 9.5 Lowering Invariants

Lowering MUST preserve architecture-observable semantics:

### 9.5.1 Valid-Region Semantics

```text
// Input: Tile with Rv=8, Cv=16 (physical 16x16)
// Lowering must preserve valid region

Virtual ISA: %t = tload %mem : (...) -> !pto.tile<16x16xf32, rv=8, cv=16>
// Valid region: [0,8) x [0,16)
// Elements outside valid region are undefined

Lowering: Must maintain same valid-region contract
// Compute only on valid region
// Output tile must have same Rv, Cv metadata
```

### 9.5.2 Ordering Dependencies

```text
// Explicit dependency must be preserved
%e0 = tload %mem_a : ...;  // Produces event
tsync %e0;                   // Establishes ordering
%r = tadd %a, %b : ...;     // Happens after sync

// Lowering must:
// 1. Emit code that loads before compute
// 2. Preserve event dependency chain
// 3. Not reorder across sync boundary
```

### 9.5.3 Operation Semantics

```text
// Operation meaning within architecture-defined domains:
// Domain: f32 elementwise addition

%r = tadd %a, %b : tile<f32> -> tile<f32>
// Semantics: r[i,j] = a[i,j] + b[i,j] for all valid i,j

// Lowering must:
// 1. Compute same mathematical result
// 2. Not change precision (unless explicit conversion)
// 3. Preserve element-wise nature
```

### 9.5.4 Implementation-Defined Behavior

Lowering MUST NOT silently reinterpret implementation-defined behavior as architecture-defined behavior:

```text
// BAD: Silently changing behavior
// Implementation-defined: TLOAD alignment preference
// Lowering MUST NOT: Assume aligned and skip checks

// GOOD: Explicit handling
// If implementation-defined behavior matters,
// lowering must either:
// 1. Preserve as-is, or
// 2. Fail with diagnostic if can't preserve
```

## 9.6 Source Alignment Rules

IR contracts MUST stay synchronized with:

### 9.6.1 Documentation Sources

| Source | Purpose |
|--------|---------|
| `docs/isa/*.md` | Semantic intent for each instruction |
| `include/pto/common/pto_instr.hpp` | API-level intrinsics |
| `docs/grammar/PTO-AS.md` | Textual assembly syntax |
| `docs/isa/manifest.yaml` | Master instruction inventory |

### 9.6.2 Synchronization Requirements

When updating IR:

1. Update IR definition
2. Update documentation
3. Update grammar (if syntax changes)
4. Update intrinsic headers (if API changes)
5. Verify manifest alignment

### 9.6.3 Consistency Check

```bash
# Verify documentation matches implementation
docs/tools/check_virtual_manual_consistency.py

# Verify manifest matches source
docs/tools/check_isa_manifest.py
```

## 9.7 Compatibility Policy

### 9.7.1 Evolution Rules

| Change Type | Policy |
|-------------|--------|
| New operation | Additive, allowed |
| New optional attribute | Additive, allowed |
| New required attribute | Breaking, requires version bump |
| Changed operation semantics | Breaking, requires version bump |
| New type | Additive, allowed |

### 9.7.2 Versioning Requirements

Breaking IR contract changes MUST include:

- Version field in IR metadata
- Migration notes explaining changes
- Compatibility mode for old versions

```mlir
// Versioned IR
module @versioned_module attributes {pto.version = "2.0"} {
  ...
}
```

### 9.7.3 Unknown Field Handling

| Field Type | Policy |
|------------|--------|
| Unknown required field | MUST reject |
| Unknown optional field | MUST reject unless compatibility mode permits |
| Unknown operation | MUST reject with deterministic error |

### 9.7.4 Deprecation Policy

Deprecated constructs SHOULD:

- Remain parseable for at least one compatibility window
- Emit deprecation warnings
- Include migration guidance in diagnostics

## 9.8 Diagnostics Requirements

### 9.8.1 Diagnostic Content

IR/verifier diagnostics MUST include:

| Requirement | Description |
|-------------|-------------|
| Operation identifier | Which operation failed |
| Location context | File, line, column |
| Expected vs actual | What was expected vs. received |
| Error class | Deterministic code for CI |

### 9.8.2 Diagnostic Classes

| Class | Description |
|-------|-------------|
| PTO-STR-001 | Invalid operation name |
| PTO-STR-002 | Operand count mismatch |
| PTO-STR-003 | Type class mismatch |
| PTO-STR-004 | Missing required attribute |
| PTO-LEGAL-001 | Unsupported dtype |
| PTO-LEGAL-002 | Unsupported shape |
| PTO-LEGAL-003 | Unsupported location-intent |

### 9.8.3 Example Diagnostics

```
Error [PTO-STR-002] at module.mlir:10:5
  Operation: tadd
  Expected: 2 operands (tile<f32>, tile<f32>)
  Received: 1 operand (tile<f32>)
  Fix: Add second operand

Error [PTO-LEGAL-002] at module.mlir:15:10
  Operation: tmatmul
  Backend: Ascend A2
  Supported shapes: 8x8, 16x16
  Received: 32x32
  Fix: Use supported shape or select different backend
```

## 9.9 Minimum Conformance Scenarios

Conformance validation SHOULD include:

### 9.9.1 Test Categories

| Category | Description |
|----------|-------------|
| Structural tests | Legal and illegal IR structures |
| Legality tests | Backend profile pass/fail matrix |
| Round-trip tests | Text -> IR -> bytecode -> IR -> Text |
| Semantic tests | Per-instruction behavior validation |

### 9.9.2 Required Test Coverage

```
Conformance Suite:
  Structural:
    - Valid operation instances
    - Invalid operation instances (negative tests)
    - Type mismatches
    - Missing attributes
    
  Legality:
    - Legal dtype/layout/location tuples
    - Illegal tuples with proper diagnostics
    
  Round-trip:
    - PTO-AS -> IR -> PTO-AS
    - IR -> bytecode -> IR
    - Full pipeline: PTO-AS -> IR -> bytecode -> IR -> text
    
  Semantic:
    - Per-instruction correctness
    - Valid-region handling
    - Synchronization ordering
```

## 9.10 Related Documentation

| Topic | Reference |
|-------|-----------|
| Assembly format | Chapter 6 |
| Bytecode | Chapter 10 |
| Instruction reference | docs/isa/*.md |
| Grammar | docs/grammar/PTO-AS.md |
| Intrinsics | include/pto/common/pto_instr.hpp |
