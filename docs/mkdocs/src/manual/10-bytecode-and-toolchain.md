# 10. Bytecode and Toolchain

This chapter defines the practical interchange and validation contract for PTO-AS, PTO IR, and bytecode forms. It specifies how these representations interact to enable reliable code generation and verification.

## 10.1 Scope

This chapter covers:

- **Representation layers**: PTO-AS, IR, and bytecode relationships
- **Bytecode module contract**: Version 1 requirements
- **Validation pipeline**: Recommended verification workflow
- **Diagnostics contract**: Error reporting standards
- **Compatibility policy**: Version evolution rules
- **Round-trip guarantees**: Text-IR-bytecode transformations

## 10.2 Representation Layers

PTO supports four representation layers that must preserve semantic equivalence:

### 10.2.1 Layer Hierarchy

```
+--------------------+     +-------------------+
|  Source Code       | --> | PTO-AS Text       |  <- Human-readable
+--------------------+     +-------------------+
                                    |
                                    v
                          +-------------------+
                          | PTO IR (MLIR)     |  <- Structured
                          +-------------------+
                                    |
                                    v
                          +-------------------+
                          | PTO Bytecode      |  <- Serialized
                          +-------------------+
```

### 10.2.2 Layer Responsibilities

| Layer | Purpose | Key Properties |
|-------|---------|---------------|
| **PTO-AS Text** | Human-readable assembly | One instruction per line, SSA names |
| **PTO IR** | Structured representation | Typed operations, control flow |
| **PTO Bytecode** | Serialized interchange | Binary format, versioned |
| **Target Code** | Executable form | Backend-specific |

### 10.2.3 Semantic Preservation

Layer transitions MUST preserve architecture-observable meaning:

```text
// Original PTO-AS
%r = tadd %a, %b : (tile<f32>, tile<f32>) -> tile<f32>;

// IR Representation
%r = "tadd"(%a, %b) : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>;

// Bytecode (serialized)
// Operation: 0x01 (TADD)
// Operands: [ref_a, ref_b]
// Result: ref_r

// Constraint: All forms must compute same result
// r[i,j] = a[i,j] + b[i,j] for valid region
```

## 10.3 Bytecode Module Contract (v1)

A conforming v1 bytecode module MUST preserve:

### 10.3.1 Required Components

| Component | Description |
|-----------|-------------|
| Operation ordering | Sequence of operations maintained |
| SSA def-use topology | Value dependencies preserved |
| Operand/result types | Type information retained |
| Required attributes | Mode, config metadata included |
| Symbol identity | Entrypoint names unchanged |

### 10.3.2 Serialization Requirements

If lossless preservation is impossible, serialization MUST fail deterministically:

```text
// GOOD: Fail with clear error
Error: Cannot serialize IR with unconnected operations
  Operation: %r = tadd %a, %b
  Issue: SSA value 'a' has no definition
  Fix: Define all SSA values before use

// BAD: Silently drop information
// (No silent information loss allowed)
```

### 10.3.3 Bytecode Format Structure

```text
PTO Bytecode v1 Format:
  +------------------+
  | Magic Number     |  0xPTO0010
  +------------------+
  | Version          |  1.0
  +------------------+
  | Module Header    |  Name, function count
  +------------------+
  | Function Table   |  Offsets to functions
  +------------------+
  | Operations       |  Serialized ops
  +------------------+
  | String Table     |  Symbol names
  +------------------+
  | Type Table       |  Type descriptions
  +------------------+
  | Attributes       |  Operation attributes
  +------------------+
  | Checksum         |  Integrity verification
  +------------------+
```

## 10.4 Validation Pipeline

### 10.4.1 Recommended Pipeline

```
1. Parse PTO-AS Text
   |
   v
2. Run Structural Verifier
   |
   v
3. Serialize IR to Bytecode
   |
   v
4. Deserialize Bytecode to IR
   |
   v
5. Re-run Structural Verifier
   |
   v
6. (Optional) Run Target Legality Verifier
```

### 10.4.2 Pipeline Stages

| Stage | Purpose | Failure Action |
|-------|---------|----------------|
| Parse | Text -> IR | Report syntax errors |
| Structural Verify | Validate IR structure | Reject malformed IR |
| Serialize | IR -> Bytecode | Fail if information loss |
| Deserialize | Bytecode -> IR | Report format errors |
| Verify Again | Confirm round-trip integrity | Fail if mismatch |
| Target Verify | Backend-specific validation | Report unsupported features |

### 10.4.3 CI Enforcement

CI SHOULD enforce steps 1-5:

```bash
# CI Pipeline
- name: Parse and Verify
  run: |
    ptoas input.pt --verify --output /dev/null

- name: Serialize
  run: |
    ptoas input.pt --output input.ptobc

- name: Deserialize
  run: |
    ptobc input.ptobc --output /tmp/ir.mlir

- name: Verify Round-trip
  run: |
    diff <(ptoas input.pt --format mlir) <(ptobc input.ptobc --format mlir)
```

## 10.5 Diagnostics Contract

### 10.5.1 Diagnostic Requirements

Diagnostics MUST be:

- **Location-aware**: Include file, line, column for textual forms
- **Deterministic**: Same input produces same error message
- **Actionable**: Include expected vs. actual information

### 10.5.2 Error Classes

| Error Class | Description |
|-------------|-------------|
| PTO-PARSE-001 | Syntax error |
| PTO-PARSE-002 | Invalid token |
| PTO-STR-001 | Structural verification error |
| PTO-STR-002 | Type mismatch |
| PTO-BC-001 | Bytecode format error |
| PTO-BC-002 | Bytecode version mismatch |
| PTO-LEGAL-001 | Target legality error |

### 10.5.3 Example Diagnostics

```text
// Parse Error
Error [PTO-PARSE-001] at program.pt:10:5
  Unexpected token: ';'
  Expected: ',' or ')' in operand list
  Context: %dst = tadd %a %b;

// Structural Error
Error [PTO-STR-001] at program.pt:15:10
  Operation: tadd
  Issue: Operand type mismatch
  Expected: tile<f32>, tile<f32>
  Received: tile<f32>, tile<i32>

// Bytecode Error
Error [PTO-BC-001] at program.ptobc:0
  Magic number mismatch
  Expected: 0xPTO0010
  Received: 0xDEADBEEF

// Target Legality Error
Error [PTO-LEGAL-001] at program.pt:20:5
  Operation: tmatmul
  Backend: Ascend A2
  Issue: Unsupported tile shape
  Supported: 8x8, 16x16
  Received: 32x32
```

## 10.6 Compatibility Policy

### 10.6.1 Evolution Policy Requirements

Evolution policy MUST define:

| Requirement | Description |
|-------------|-------------|
| Schema version | Version field in bytecode header |
| Backward compatibility | Minimum supported version |
| Unknown-field handling | Policy for unrecognized fields |
| Unknown-op handling | Policy for unrecognized operations |

### 10.6.2 Default Policies

| Scenario | Policy |
|----------|--------|
| Unknown required field | Reject |
| Unknown optional field | Reject (unless explicit compatibility mode) |
| Unknown operation | Reject with deterministic error |
| Version too old | Reject with migration hint |

### 10.6.3 Version Negotiation

```text
// Toolchain version negotiation
Toolchain version: 2.0
Bytecode version: 1.5

// Compatibility check
If toolchain.version >= bytecode.version:
  // Compatible
Else:
  Error: Bytecode version too new
  Hint: Upgrade toolchain to version X or later
```

### 10.6.4 Migration Support

When bytecode format changes:

- Provide migration tool
- Document breaking changes
- Support old format for at least one release cycle

## 10.7 Round-Trip Guarantees

### 10.7.1 Guaranteed Preservation

For supported features, `text -> IR -> bytecode -> IR -> text` SHOULD preserve:

| Property | Guarantee |
|----------|-----------|
| Semantics | Computed results identical |
| Verifier-relevant structure | All validations pass |
| Required metadata | Attributes, types preserved |
| Operation ordering | Sequence maintained |

### 10.7.2 Non-Required Preservation

Byte-for-byte textual formatting equivalence is NOT required:

```text
// Original
%dst = tadd %src0, %src1 : (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> !pto.tile<16x16xf32>;

// After round-trip (format may differ)
%dst = tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>;

// Both are semantically equivalent
```

### 10.7.3 Round-Trip Verification

```bash
# Verify round-trip
ptoas input.pt --output /tmp/ir.mlir
ptobc /tmp/ir.mlir --output input.ptobc
ptobc input.ptobc --output /tmp/ir2.mlir
diff /tmp/ir.mlir /tmp/ir2.mlir
# Must be identical after normalization
```

## 10.8 Operational Acceptance Checklist

Each release SHOULD validate:

### 10.8.1 Required Test Suites

| Suite | Description |
|-------|-------------|
| Parser positive | Valid PTO-AS parses successfully |
| Parser negative | Invalid PTO-AS produces errors |
| Structural verifier | Valid IR passes verification |
| Structural negative | Invalid IR rejected with errors |
| Malformed bytecode | Corrupted bytecode handled gracefully |
| Round-trip corpus | Representative programs survive round-trip |
| Diagnostic stability | Error messages consistent |

### 10.8.2 Example Test Cases

```text
// Parser Positive Test
Input: "%r = tadd %a, %b : (tile<f32>, tile<f32>) -> tile<f32>;"
Expected: Parse success, IR generated

// Parser Negative Test
Input: "%r = tadd %a : (tile<f32>) -> tile<f32>;"
Expected: Parse error - missing operand

// Structural Positive Test
Input: Valid IR with correct types
Expected: Verification passes

// Structural Negative Test  
Input: IR with type mismatch
Expected: Verification fails with PTO-STR-002

// Malformed Bytecode Test
Input: Bytecode with corrupted checksum
Expected: Rejected with PTO-BC-001

// Round-trip Test
Input: Representative PTO program
Expected: Output IR identical to input IR (normalized)
```

## 10.9 Related Documentation

| Topic | Reference |
|-------|-----------|
| Assembly format | Chapter 6 |
| Virtual ISA and IR | Chapter 9 |
| Instruction reference | docs/isa/*.md |
| Grammar | docs/grammar/PTO-AS.md |
| Bytecode format | docs/ir/PTO-IR-bytecode.md |
