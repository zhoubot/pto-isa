# 11. Memory Ordering and Consistency

This chapter defines architecture-visible memory ordering and visibility guarantees for PTO Virtual ISA operations.

## 11.1 Scope

This chapter covers:

- **Memory domains**: Architecture-visible memory regions
- **Consistency baseline**: Dependency-ordered consistency model
- **Ordering guarantees**: Visibility requirements for operations
- **Implementation-defined behavior**: Backend flexibility boundaries
- **Programming requirements**: Guidelines for correct programs
- **Conformance tests**: Validation requirements

## 11.2 Memory Objects and Domains

### 11.2.1 Architecture-Visible Memory Domains

Architecture-visible memory domains include:

| Domain | Description | Visibility |
|--------|-------------|------------|
| **Tile-local values** | Registers in tile storage | Core only |
| **Global memory views** | GM accessed by TLOAD/TSTORE | Host and Core |
| **Synchronization state** | Events affecting visibility | Host, Device, Core |

### 11.2.2 Memory Hierarchy

```
+-------------------+
|   Host Memory     |  <- External to PTO Core
+-------------------+
         |
         v
+-------------------+
| Global Memory     |  <- GM (visible to PTO Core)
| (memref)          |
+-------------------+
         |
         v
+-------------------+
| Tile Local        |  <- In-core storage
| (tile<...>)       |
+-------------------+
```

### 11.2.3 Backend Caches

Backend-private caches/buffers are implementation-defined, but MUST respect architecture-visible ordering outcomes:

```text
// Implementation-defined: Cache behavior
Backend may:
// - Cache tile data in local buffer
// - Prefetch data ahead of use
// - Write-back dirty data

But must:
// - Preserve explicit ordering (TSYNC)
// - Not violate producer-consumer visibility
// - Maintain valid-region semantics
```

## 11.3 Consistency Baseline

### 11.3.1 Dependency-Ordered Consistency

The baseline model is dependency-ordered consistency:

- **Data dependencies** define required visibility order
- **Explicit synchronization** defines visibility boundaries
- **Independent operations** MAY be reordered internally
- **Synchronization points** MUST establish visibility as specified

### 11.3.2 Ordering Model

```
Program Order                    Visibility Order
+---------------+                +---------------+
| %e0 = tload A |                | %e0 = tload A |
| tsync %e0     |                | tsync %e0     |  <- Barrier
+---------------+                +---------------+
         |                                |
         v                                v
+---------------+                +---------------+
| %r = tadd B,C |                | %r = tadd B,C |  <- Sees A
+---------------+                +---------------+
```

### 11.3.3 Independent Operation Reordering

Independent operations MAY be reordered:

```text
// Independent operations (no dependency)
%t0 = tload %mem0 : ...;
%t1 = tload %mem1 : ...;

// Legal reorderings:
// - t0, t1 (original)
// - t1, t0 (reordered - legal because independent)

// NOT independent (has dependency)
%t0 = tload %mem0 : ...;
tsync %t0;
%t1 = tadd %t0, %t1 : ...;
// Must maintain: t0 -> sync -> t1
```

## 11.4 Ordering Guarantees

### 11.4.1 Producer-Consumer Guarantees

A conforming implementation MUST ensure:

- **Producer writes** become visible to dependent consumers after required synchronization
- **Memory operations** participating in explicit dependency chains preserve those chains
- **TSYNC and event dependencies** are reflected in memory visibility

### 11.4.2 Explicit Dependency Chain

```text
// Explicit dependency chain
1. Producer: %e0 = tload %mem_a : ...;
//    Writes data to tile from memory_a

2. Sync: tsync %e0;
//    Establishes visibility boundary

3. Consumer: %r = tadd %tile_a, %tile_b : ...;
//    Guaranteed to see data from load
```

### 11.4.3 Event Semantics

Events establish ordering:

```text
// Event carries dependency
%e0 = tload %mem0 : ... -> !pto.tile<...>;
// %e0 represents: data is available

tsync %e0;
// After sync: all prior operations visible

%r = tadd %a, %b : ...;
// Guaranteed to see %mem0 data
```

## 11.5 Unspecified and Implementation-Defined Behavior

### 11.5.1 Architecture-Restricted Behavior

| Category | Description | Policy |
|----------|-------------|--------|
| **Out-of-domain access** | Reading undefined tile regions | Unspecified |
| **Timing details** | Cycle-accurate timing | Implementation-defined |
| **Cache policy** | Prefetch, write-back decisions | Implementation-defined |
| **Backend optimization** | Reordering that preserves visibility | Allowed |

### 11.5.2 Out-of-Domain Semantics

```text
// Tile with valid region (Rv=8, Cv=16)
// Physical tile: 16x16

// Specified region: [0,8) x [0,16)
// Elements here have defined values

// Unspecified region: [8,16) x [0,16)
// Elements here have UNDEFINED values

// UNDEFINED BEHAVIOR:
%tile = tload %mem : (10x10) -> tile<16x16>;
// Reading elements beyond row 10 is undefined
```

### 11.5.3 Implementation-Defined Flexibility

```text
// Implementation-defined: Allowed flexibility
// (Backend-specific, must preserve visible behavior)

Backend MAY:
// - Reorder independent loads/stores
// - Combine adjacent memory operations
// - Prefetch based on access patterns
// - Use vectorized memory operations

Backend MUST NOT:
// - Reorder across sync boundary
// - Make un-synced data visible early
// - Violate explicit dependency chain
```

## 11.6 Programming Requirements

### 11.6.1 Explicit Synchronization

Programs SHOULD use explicit synchronization at producer/consumer boundaries:

```text
// RECOMMENDED: Explicit sync
%e0 = tload %mem_a : ...;
tsync %e0;
%r = tadd %tile_a, %tile_b : ...;

// NOT RECOMMENDED: Implicit ordering assumed
%e0 = tload %mem_a : ...;
// No sync - order undefined
%r = tadd %tile_a, %tile_b : ...;
```

### 11.6.2 Dependency Declaration

Avoid assuming implicit global ordering without a defined dependency:

```text
// BAD: Assuming implicit ordering
// Two loads with no declared dependency
%t0 = tload %mem0 : ...;
%t1 = tload %mem1 : ...;
// Order is unspecified

// GOOD: Explicit dependency if needed
%e0 = tload %mem0 : ...;
tsync %e0;
%e1 = tload %mem1 : ...;
tsync %e1;
// Order is explicit
```

### 11.6.3 Valid-Region Usage

Avoid relying on unspecified out-of-domain values:

```text
// BAD: Using undefined values
%tile = tload %mem : (10x10) -> tile<16x16>;
// Only rows 0-9 are valid
// Using rows 10-15 is undefined

// GOOD: Using only valid region
%tile = tload %mem : (16x16) -> tile<16x16>;
// All rows are valid
// Or: Track Rv, Cv and use only valid elements
```

### 11.6.4 Manual Mode Requirements

Manual mode programmers MUST ensure required ordering when tool-managed synchronization is not used:

```text
// Manual mode: Full programmer responsibility

.arg %lhs : !pto.tile<...>;
.arg %rhs : !pto.tile<...>;

// Programmer MUST:
// 1. Declare dependencies explicitly
// 2. Issue sync before consumption
// 3. Ensure data is ready before use

%e_lhs = tload %mem_lhs : ...;
tsync %e_lhs;
%e_rhs = tload %mem_rhs : ...;
tsync %e_rhs;
%result = tmatmul %acc, %lhs, %rhs : ...;
```

## 11.7 Diagnostics and Conformance Tests

### 11.7.1 Backend Diagnostics

Backends SHOULD provide diagnostics for:

| Diagnostic | Description |
|------------|-------------|
| Missing ordering | Sync required but not present |
| Unsupported ordering | Backend doesn't support requested memory order |
| Profile restriction | Specific ordering not available in profile |

### 11.7.2 Example Diagnostics

```text
// Missing synchronization
Error [PTO-ORDER-001] at program.pt:15
  Operation: tadd
  Issue: Uses tile from load without synchronization
  Tile: %tile_0 (from line 10)
  Fix: Add tsync between load and use

// Unsupported memory order
Error [PTO-ORDER-002] at program.pt:20
  Operation: tload
  Backend: CPU Simulator
  Issue: Requested memory order not supported
  Requested: strong
  Supported: relaxed
```

### 11.7.3 Conformance Test Requirements

Conformance tests SHOULD include ordered visibility scenarios:

```text
Test Categories:
  +---------------------------+
  | Producer-Consumer         |  Load -> Sync -> Use
  +---------------------------+
  | Write-After-Read          |  Read -> Write (same location)
  +---------------------------+
  | Read-After-Write          |  Write -> Read (same location)
  +---------------------------+
  | Write-After-Write        |  Write -> Write (same location)
  +---------------------------+
  | Independent Reordering   |  No dependency - any order OK
  +---------------------------+
```

### 11.7.4 Test Matrix

| Pattern | Code | Expected Behavior |
|---------|------|-------------------|
| PAR | Load A -> Sync -> Use A | Use sees A |
| WAR | Use A -> Sync -> Load B | Load sees A written |
| WAW | Write A -> Sync -> Write B | B overwrites A |
| Independent | Load A, Load B | Any order acceptable |

## 11.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Synchronization | Chapter 5 |
| Valid-region semantics | Chapter 3, Chapter 4 |
| Memory operations | Chapter 7, TLOAD/TSTORE |
| Backend profiles | Chapter 12 |
