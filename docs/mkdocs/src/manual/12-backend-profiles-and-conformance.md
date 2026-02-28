# 12. Backend Profiles and Conformance

This chapter defines how backend capability subsets are described and how conformance levels are evaluated.

## 12.1 Scope

This chapter covers:

- **Backend profile model**: Capability documentation structure
- **Capability gating**: Toolchain enforcement of profile support
- **Conformance dimensions**: Evaluation criteria
- **Conformance levels**: Tiered validation stages
- **Test matrix**: Required test coverage
- **Change management**: Update procedures

## 12.2 Backend Profile Model

### 12.2.1 Profile Documentation Requirements

A backend profile MUST document:

| Requirement | Description |
|-------------|-------------|
| Instruction families | Supported operations |
| Supported tuples | dtype/layout/location/shape combinations |
| Synchronization limits | Memory ordering support |
| Implementation-defined | Backend-specific behavior surface |
| Diagnostics policy | Error reporting for unsupported features |

### 12.2.2 Profile Examples

Profiles MAY correspond to concrete targets:

| Profile | Target | Description |
|---------|--------|-------------|
| A2 | Ascend 910 | High-end AI accelerator |
| A3 | Ascend 910B | Enhanced AI accelerator |
| A5 | Ascend 910Pro | Premium AI accelerator |
| CPU | CPU Simulator | Software emulation |

### 12.2.3 Profile Document Structure

```text
Backend Profile: Ascend A2-v1.0
========================================

1. Supported Instructions
   - Synchronization: TSYNC, Event
   - Elementwise: TADD, TSUB, TMUL, TCMP, ...
   - Memory: TLOAD, TSTORE
   - Matrix: TMATMUL (16x16, f32/f16)
   ...

2. Supported Tuples
   - Tile shapes: 8x8, 16x16
   - Element types: f32, f16, i32, i16
   - Layout classes: Vec, Mat, Left, Right, Acc
   - Location intents: per instruction

3. Synchronization
   - Event support: Yes
   - TSYNC: Yes
   - Pipeline barriers: Yes
   
4. Implementation-Defined
   - TLOAD alignment: 16-byte preferred
   - TMATMUL latency: 8-12 cycles
   - Event recycling: After TSYNC
   
5. Diagnostics
   - Unsupported tuple: PTO-LEGAL-001
   - Missing sync: PTO-ORDER-001
```

## 12.3 Capability Gating

### 12.3.1 Toolchain Requirements

Toolchains MUST gate backend-specific specialization by declared profile capability:

- Validate requested operations against profile
- Reject or fallback for unsupported features
- Produce deterministic diagnostics

### 12.3.2 Handling Unsupported Behavior

If requested behavior is outside profile support:

- **Compilation/legalization MUST fail deterministically**, or
- **An explicitly defined fallback path MUST be selected**

### 12.3.3 Example: Unsupported Tile Shape

```text
// Requested: TMATMUL with 32x32 tile
// Backend: Ascend A2 (supports 8x8, 16x16)

// Option 1: Fail deterministically
Error [PTO-LEGAL-002] at program.pt:10
  Operation: tmatmul
  Backend: Ascend A2
  Requested shape: 32x32
  Supported shapes: 8x8, 16x16
  Fix: Use 16x16 or select different backend

// Option 2: Fallback (if auto-mode)
Info: Shape 32x32 not supported, falling back to 16x16
// Toolchain substitutes 16x16
```

### 12.3.4 Example: Unsupported Data Type

```text
// Requested: TADD with f64
// Backend: Ascend A2 (supports f32, f16, i32)

// Option 1: Fail
Error [PTO-LEGAL-001] at program.pt:15
  Operation: tadd
  Backend: Ascend A2
  Requested dtype: f64
  Supported dtypes: f32, f16, i32

// Option 2: Fallback with warning
Warning: f64 not supported, using f32 (precision loss possible)
%r = tadd %a, %b : tile<f32>
```

## 12.4 Conformance Dimensions

Conformance is evaluated along these dimensions:

| Dimension | Description | Validation Method |
|-----------|-------------|------------------|
| **Semantic conformance** | Instruction behavior matches spec | Test against reference |
| **Legality conformance** | Contract validation passes | Profile tuple tests |
| **Ordering conformance** | Sync and memory visibility | Dependency tests |
| **Diagnostic conformance** | Errors are deterministic | Regression tests |

### 12.4.1 Semantic Conformance

```text
// Verify: TMATMUL computes correct result
Input: 
  Acc = zeros(16x16, f32)
  LHS = ones(16x16, f16)
  RHS = twos(16x16, f16)
Expected:
  Result[i,j] = sum(LHS[i,k] * RHS[k,j]) for k in 0..16
  = 16 * 2 = 32 (as f32)
// Verify: Actual output matches
```

### 12.4.2 Legality Conformance

```text
// Verify: All legal tuples accepted
Operation: TADD
Legal tuples: (f32, f32)->f32, (f16, f16)->f16, (i32, i32)->i32
// Test: Each tuple should pass verification

// Verify: All illegal tuples rejected
Operation: TADD  
Illegal tuples: (f32, f16)->f32, (f64, f64)->f64
// Test: Each tuple should fail with diagnostic
```

### 12.4.3 Ordering Conformance

```text
// Verify: Synchronization establishes visibility
// Sequence: Load A -> Sync -> Compute -> Sync -> Store B
// Result: Store B must see Load A's data
```

### 12.4.4 Diagnostic Conformance

```text
// Verify: Same input produces same error
Input: Invalid IR
Run 1: Error [PTO-LEGAL-001] ...
Run 2: Error [PTO-LEGAL-001] ...
// Must be identical
```

## 12.5 Conformance Levels

### 12.5.1 Level Definitions

Recommended levels:

| Level | Name | Description | Requirements |
|-------|------|-------------|--------------|
| **Level 0** | Parse/Shape | Structural correctness | Parser works, types parse |
| **Level 1** | Family Legality | Contract validation | Family rules enforced |
| **Level 2** | Instruction Semantic | Per-op validation | Behavior validated |
| **Level 3** | Cross-layer Stability | Full pipeline | All dimensions stable |

### 12.5.2 Level Details

**Level 0 - Parse/Shape**
- Parser accepts valid PTO-AS
- Parser rejects syntactically invalid input
- Types are correctly represented

**Level 1 - Family Legality**
- All family-level constraints validated
- Diagnostic messages for illegal use
- Profile-specific rules enforced

**Level 2 - Instruction Semantic**
- Per-instruction semantics verified
- Reference implementation matches
- Edge cases tested

**Level 3 - Cross-layer Stability**
- IR/bytecode/text round-trips work
- Ordering preserved across layers
- Diagnostics stable across versions

### 12.5.3 Publishing Conformance

A backend SHOULD publish the highest validated level and known gaps:

```text
Backend Profile: Ascend A2
========================================
Conformance Level: Level 2 (Instruction Semantic)

Achieved:
  - Level 0: Yes
  - Level 1: Yes
  - Level 2: Yes
  
Known Gaps:
  - Level 3: Partial (IR->bytecode works, text round-trip has formatting differences)
  
Roadmap:
  - Level 3: Target v2.1
```

## 12.6 Required Test Matrix

### 12.6.1 Test Categories

A profile conformance suite SHOULD include:

| Category | Description |
|----------|-------------|
| Legal tuple tests | Valid dtype/layout/location/shape by instruction |
| Illegal tuple tests | Invalid combinations rejected |
| Ordering tests | Sync and memory visibility |
| Precision tests | Mode interactions, mixed precision |
| Round-trip tests | Text/IR/bytecode transformations |
| Diagnostic tests | Error message stability |

### 12.6.2 Test Matrix Example

| Instruction | dtypes | Shapes | Locations | Expected |
|-------------|--------|--------|------------|----------|
| TADD | f32, f16 | 8x8, 16x16 | Vec, Mat | Legal |
| TADD | f64 | 32x32 | Acc | Illegal |
| TMATMUL | f32xf16->f32 | 16x16 | Left, Right, Acc | Legal |
| TMATMUL | f64xf64->f64 | 32x32 | Any | Illegal |
| TLOAD | f32 | 8x8, 16x16 | - | Legal |
| TLOAD | f80 | 64x64 | - | Illegal |

### 12.6.3 Ordering Test Suite

```
Test Suite: Synchronization Ordering
===================================

Test 1: Producer-Consumer
  Setup: Load -> Sync -> Compute -> Store
  Verify: Store sees Load data

Test 2: RAW Hazard
  Setup: Read A -> Write A (no sync)
  Verify: Illegal (or sync required)

Test 3: WAR Hazard  
  Setup: Write A -> Read A (no sync)
  Verify: Illegal (or sync required)

Test 4: WAW Hazard
  Setup: Write A -> Write A (no sync)
  Verify: Illegal (or sync required)

Test 5: Independent Reordering
  Setup: Load A, Load B (no dependency)
  Verify: Any order acceptable
```

## 12.7 Change Management

### 12.7.1 Update Requirements

When backend behavior changes:

- **Profile documents MUST be updated** in the same change set
- **Conformance impact MUST be stated**
- **Regressions against published levels** MUST be treated as release blockers unless explicitly waived with rationale

### 12.7.2 Change Classification

| Change Type | Impact | Action |
|-------------|--------|--------|
| New instruction | Add to profile | Update docs, tests |
| New dtype support | Extend profile | Update docs, tests |
| New shape support | Extend profile | Update docs, tests |
| Bug fix (behavior) | May change conformance | Re-validate level |
| Performance change | No conformance impact | Document |

### 12.7.3 Release Notes Template

```text
Backend Changes: Ascend A2-v1.1
===================================

New Features:
  - Added support for TMATMUL with f32xTF32->f32
  - Added support for 32x32 tiles on TADD

Breaking Changes:
  - TLOAD alignment requirement changed from 8-byte to 16-byte
  - Impact: Level 2 conformance maintained, Level 3 affected

Bug Fixes:
  - Fixed TMAX with NaN handling (was incorrect, now matches spec)
  - Impact: Re-validated Level 2

Conformance Status:
  - Level 0: Yes
  - Level 1: Yes
  - Level 2: Yes
  - Level 3: Yes (except text round-trip formatting)
```

### 12.7.4 Waivers

A regression waiver MUST include:

- Description of regression
- Root cause analysis
- Impact assessment
- Remediation plan
- Timeline for fix

## 12.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Virtual ISA | Chapter 1-11 |
| Instruction reference | docs/isa/*.md |
| Profiles | Backend-specific documents |
| Diagnostics | Appendix C |
| Testing | docs/testing/*.md |
