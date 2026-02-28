# 8. Programming Model Contracts

This chapter defines architecture-safe programming contracts for Auto and Manual modes. It focuses on correctness and portability boundaries rather than backend-specific optimization tricks.

## 8.1 Scope

This chapter covers:

- **Auto vs Manual mode**: Contract split between toolchain and programmer responsibilities
- **Portability rules**: Safe programming practices for cross-backend code
- **Performance patterns**: Portable optimization strategies
- **Anti-patterns**: Non-portable behaviors to avoid
- **Debug workflow**: Recommended validation pipeline
- **Compatibility**: Handling implementation-defined behavior

## 8.2 Auto and Manual Contract Split

PTO supports two programming models that differ in responsibility distribution:

### 8.2.1 Auto Mode

In Auto mode, the toolchain manages resource allocation and synchronization:

| Responsibility | Toolchain Action |
|----------------|------------------|
| Tile placement | Infers legal tile-to-address bindings |
| Ordering | Infers synchronization points from dataflow |
| Scheduling | Determines execution order for efficiency |

**Contract Requirements:**

- Toolchain SHOULD infer legal placement, ordering, and scheduling
- Generated code MUST preserve Virtual ISA semantics
- User-visible behavior MUST remain deterministic under equivalent source and options

**Example Auto Mode:**

```python
# PTO-DSL (Auto mode)
@pto.program
def matmul(lhs, rhs):
    # Toolchain infers:
    # - Tile sizes based on target
    # - Memory addresses for tiles
    # - Synchronization points
    return ptodsl.matmul(lhs, rhs)
```

### 8.2.2 Manual Mode

In Manual mode, the programmer explicitly controls resources and synchronization:

| Responsibility | Programmer Action |
|----------------|-------------------|
| Tile placement | Explicit tile-to-address bindings |
| Ordering | Explicit TSYNC/event dependencies |
| Scheduling | Manual program structuring |

**Contract Requirements:**

- Programmers MAY explicitly control placement and synchronization
- User-authored dependencies and ordering points MUST be preserved
- Illegal manual configurations MUST fail with actionable diagnostics

**Example Manual Mode (PTO-AS):**

```text
// Manual mode - explicit resource binding
.arg %lhs : !pto.tile<16x16xf16>;
.arg %rhs : !pto.tile<16x16xf16>;
.arg %acc : !pto.tile<16x16xf32>;

// Explicit tile assignment
tassign %tile0, %lhs;
tassign %tile1, %rhs;
tassign %tile2, %acc;

// Explicit synchronization
%e0 = tload %mem0 : ...;
tsync %e0;

%e1 = tload %mem1 : ...;
tsync %e1;

// Compute with explicit ordering
%r = tmatmul %tile2, %tile0, %tile1 : ...;
tsync %r;

tstore %mem2, %r : ...;
```

### 8.2.3 Mode Comparison Table

| Aspect | Auto Mode | Manual Mode |
|--------|-----------|-------------|
| Resource binding | Toolchain-inferred | Programmer-specified |
| Synchronization | Dataflow-driven | Explicit TSYNC/events |
| Scheduling | Toolchain-optimized | Programmer-controlled |
| Portability | Higher | Lower (profile-dependent) |
| Error recovery | Toolchain fallback | Programmer fix required |

## 8.3 Portability-Safe Programming Rules

Programs intended for cross-backend portability SHOULD:

### 8.3.1 Family-Level Legality

```text
// SAFE: Use documented family-level operations
%dst = tadd %a, %b : ...;  // Elementwise family

// UNSAFE: Rely on implementation-specific behavior
%dst = tadd %a, %b {implementation_specific_attr = ...} : ...;
```

### 8.3.2 Explicit Synchronization

```text
// SAFE: Explicit producer/consumer boundaries
%e0 = tload %mem0 : ... -> !pto.tile<...>;
tsync %e0;  // Explicit dependency
%r = tadd %a, %b : ...;

// UNSAFE: Assuming implicit ordering
%e0 = tload %mem0 : ... -> !pto.tile<...>;
// No sync - ordering undefined
%r = tadd %a, %b : ...;
```

### 8.3.3 Backend Intersection Profiles

```text
// SAFE: Use tuples in backend intersection
// If A2 supports {f32, 16x16} and A3 supports {f16, 32x32}
// Use intersection or profile-gated code

// SAFE: Capability detection
.arg %tile : !pto.tile<...>;
.const %cap = query_capability : ...;
tsel %out, %tile, %fallback {mask = %cap};
```

## 8.4 Performance-Aware but Portable Patterns

### 8.4.1 Domain-Safe Tiling

```python
# Portable tiling pattern
def tile_matmul(A, B, tile_size=16):
    M, K = A.shape
    K, N = B.shape
    
    # Pad to tile boundary for full tiles
    M_padded = ((M + tile_size - 1) // tile_size) * tile_size
    N_padded = ((N + tile_size - 1) // tile_size) * tile_size
    
    # Process full tiles only (valid-region safe)
    for i in range(0, M_padded, tile_size):
        for j in range(0, N_padded, tile_size):
            # Valid region: min(tile_size, remaining)
            rv = min(tile_size, M - i)
            cv = min(tile_size, N - j)
            process_tile(A, B, i, j, rv, cv)
```

### 8.4.2 Clear Phase Boundaries

```text
// SAFE: Clear producer/consumer with events
// Phase 1: Load
%e_load0 = tload %mem_a : ... -> !pto.tile<...>;
%e_load1 = tload %mem_b : ... -> !pto.tile<...>;

// Phase 2: Sync before compute
tsync %e_load0;
tsync %e_load1;

// Phase 3: Compute
%e_comp = tmatmul %acc, %tile_a, %tile_b : ...;

// Phase 4: Sync before store
tsync %e_comp;

// Phase 5: Store
tstore %mem_out, %result : ...;
```

### 8.4.3 Backend-Gated Specialization

```text
// SAFE: Profile-gated code
.arg %arg : !pto.tile<16x16xf32>;

// Check backend capability
.const %has_tf32 = query_backend_capability {cap = TF32_MATMUL};

tsel %out, %tile_tf32, %tile_f32 {mask = %has_tf32};
```

## 8.5 Anti-Patterns

### 8.5.1 Out-of-Domain Values

```text
// ANTI-PATTERN: Reading undefined regions
%tile = tload %mem : (!pto.memref<10x10xf32>) -> !pto.tile<16x16xf32>;
// Tile has 10x10 valid, 6x6 undefined

// DO NOT: Use undefined region as meaningful data
%result = tadd %tile, %other : ...;  // Undefined behavior
```

### 8.5.2 Pipeline Timing Dependencies

```text
// ANTI-PATTERN: Depending on undocumented timing
%e0 = tload %mem : ...;
%e1 = tadd %a, %b : ...;
// DO NOT: Assume e1 happens after e0 without sync

// CORRECT: Explicit synchronization
%e0 = tload %mem : ...;
tsync %e0;
%e1 = tadd %a, %b : ...;
```

### 8.5.3 Implicit Ordering Assumptions

```text
// ANTI-PATTERN: Assuming implicit dependency
%tile_a = tload %mem_a : ...;
%tile_b = tload %mem_b : ...;
%result = tmatmul %acc, %tile_a, %tile_b : ...;
// DO NOT: Assume tile_b available after tile_a

// CORRECT: Explicit events if order matters
%e_a = tload %mem_a : ...;
tsync %e_a;
%e_b = tload %mem_b : ...;
tsync %e_b;
%result = tmatmul %acc, %tile_a, %tile_b : ...;
```

### 8.5.4 Unprofile-Gated Assumptions

```text
// ANTI-PATTERN: Backend-specific without gating
.arg %tile : !pto.tile<16x16xf16>;

// DO NOT: Assume specific backend behavior
%result = tadd %tile, %tile : ... {backend_specific = 1};

// CORRECT: Profile-gated
.const %profile = get_backend_profile : ... -> i32;
tsel %safe, %tile, %fallback {condition = %profile};
```

## 8.6 Debug and Validation Workflow

### 8.6.1 Recommended Pipeline

| Stage | Check | Tool |
|-------|-------|------|
| 1 | Structural correctness | Parser, type checker |
| 2 | Legal domain | Shape/layout/location validator |
| 3 | Synchronization | Dependency analyzer |
| 4 | Backend conformance | Profile validator |
| 5 | Differential behavior | Cross-target testing |

### 8.6.2 Structural Correctness Checks

```text
// Verify:
// - Operand types match instruction contract
// - Required attributes present
// - Result arity correct

%dst = tadd %src0, %src1 : 
  (!pto.tile<16x16xf32>, !pto.tile<16x16xf32>) -> 
  !pto.tile<16x16xf32>;
// Checks:
//   [PASS] Operand count: 2
//   [PASS] Operand types: tile<...>
//   [PASS] Result type: tile<...>
```

### 8.6.3 Legal Domain Checks

```text
// Verify:
// - Shape tuples legal for backend
// - Location-intent compatible with operation
// - Layout class supported

%dst = tmatmul %acc, %lhs, %rhs : 
  (!pto.tile<16x16xf32>, !pto.tile<16x16xf16>, !pto.tile<16x16xf16>) ->
  !pto.tile<16x16xf32>;
// Checks:
//   [PASS] dtype: f32 x f16 -> f32 (legal)
//   [PASS] Shape: 16x16 (legal for TMATMUL)
//   [PASS] Location: Acc, Left, Right (legal)
```

### 8.6.4 Synchronization Checks

```text
// Verify:
// - All data dependencies have explicit sync
// - No RAW/WAR/WAW hazards without sync

%e0 = tload %mem_a : ...;
%e1 = tload %mem_b : ...;
tsync %e0;
tsync %e1;
%r = tmatmul %acc, %a, %b : ...;
// Checks:
//   [PASS] RAW hazard: A,B -> compute (synced)
//   [PASS] No undefined ordering
```

## 8.7 Compatibility Notes

### 8.7.1 Documentation Requirements

When code relies on implementation-defined behavior:

- Assumptions MUST be documented
- Backend profile constraints MUST be declared
- Fallback behavior SHOULD be provided where feasible

### 8.7.2 Implementation-Defined Documentation Template

```text
// Implementation-Defined Behavior Documented
//
// Backend: Ascend A2
// Profile: A2-v1.0
//
// The following behavior is implementation-defined:
// - TMATMUL latency: 8-12 cycles (varies by tile shape)
// - TLOAD alignment: 16-byte preferred, unaligned allowed
// - Event recycling: Events may be reused after TSYNC
//
// Fallback:
// If A2-specific behavior needed, use capability detection:
//   .const %is_a2 = query_backend {profile = A2}
//   tsel %out, %a2_path, %generic_path {mask = %is_a2}
```

### 8.7.3 Profile Gating Example

```python
# Profile-gated implementation
def matmul_profile_gated(lhs, rhs):
    profile = get_backend_profile()
    
    if profile == "A2":
        # A2-specific: f32 accumulator, 16x16 tiles
        return matmul_a2(lhs, rhs)
    elif profile == "A3":
        # A3-specific: f16 with TF32 support
        return matmul_a3(lhs, rhs)
    else:
        # Generic: baseline compatibility
        return matmul_generic(lhs, rhs)
```

## 8.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Auto mode | Chapter 2, Machine Model |
| Manual mode | Chapter 2, Machine Model |
| Synchronization | Chapter 5 |
| Tile operations | Chapter 4 |
| Backend profiles | Chapter 12 |
| Diagnostics | Appendix C |
