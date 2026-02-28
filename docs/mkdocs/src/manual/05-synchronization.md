# 5. Synchronization

This chapter defines architecture-visible synchronization and ordering behavior for PTO Virtual ISA programs.

## 5.1 Scope

This chapter covers:

- **Synchronization primitives**: Events and TSYNC
- **TSYNC contract**: Ordering semantics
- **Hazard classes**: RAW, WAR, WAW
- **Event model**: Dependency management
- **Auto vs Manual responsibilities**: When each mode requires explicit sync

## 5.2 Synchronization Primitives

PTO provides the following synchronization primitives:

### 5.2.1 Event-Based Dependency Chaining

Events are explicit dependency tokens that track completion of operations:

```cpp
Event e;                              // Create event
TLOAD(tile, gmem, e);                // Associate event with operation
TSYNC(e);                            // Wait for event
```

Events provide fine-grained dependency tracking between operations.

### 5.2.2 TSYNC - Synchronization Barrier

`TSYNC` establishes ordering between operation sets:

```cpp
TSYNC();                             // Barrier: wait for all prior operations
TSYNC(e);                           // Wait for specific event
```

### 5.2.3 Pipeline Barriers

PTO supports pipeline-level synchronization:

| Primitive | Description |
|-----------|-------------|
| `TSYNC` | Full synchronization barrier |
| Event-based | Wait for specific operations |
| Implicit (Auto mode) | Compiler-inserted synchronization |

### 5.2.4 Backend Primitives

PTO abstracts backend-specific low-level primitives through architecture semantics. The following remain implementation-defined:

- Hardware-specific synchronization instructions
- Memory fence implementations
- Cache coherence protocols

## 5.3 TSYNC Contract

### 5.3.1 Ordering Guarantees

`TSYNC` establishes ordering between operation sets. A conforming implementation MUST ensure that:

1. **Operations ordered-before** the synchronization point become visible to **ordered-after** consumers according to the memory model

2. **Synchronization semantics are preserved** through optimization and lowering

3. **Unsupported synchronization forms** are rejected with deterministic diagnostics

### 5.3.2 TSYNC Semantics

```
Program Order:
    Op A (produces event e)
         |
         v
    Op B (produces event e)
         |
         v
    TSYNC(e)           <- Synchronization point
         |
         v
    Op C (consumes result)
```

All operations that produce event `e` must complete before any operation after `TSYNC(e)` can see their results.

### 5.3.3 TSYNC Variants

```cpp
// Wait for specific event
TSYNC(e0);

// Wait for multiple events (all must complete)
TSYNC(e0);
TSYNC(e1);

// Wait for any event (implementation-defined which)
TSYNC(e0);
TSYNC(e1);
// Note: Behavior may vary by backend
```

## 5.4 Hazard Classes

Synchronization requirements commonly arise from data hazards in pipelined execution:

### 5.4.1 Read-After-Write (RAW) Hazards

Also known as "true data dependencies":

```
Core 0:                    Core 1:
  STORE tile0 ------> READ tile0
   (write)              (read)
```

**Solution**: Use event + TSYNC to ensure store completes before read

```cpp
Event e;
TSTORE(gmem, tile0, e);    // Store produces event
TSYNC(e);                   // Wait for store
TLOAD(tile1, gmem);         // Now safe to read
```

### 5.4.2 Write-After-Read (WAR) Hazards

Also known as "anti-dependencies":

```
Core 0:                    Core 1:
  READ tile0 <------- WRITE tile0
   (read)              (write)
```

**Solution**: Ensure write happens after read completes

### 5.4.3 Write-After-Write (WAW) Hazards

Also known as "output dependencies":

```
Core 0:                    Core 1:
  WRITE tile0 -----> WRITE tile0
   (write 1)           (write 2)
```

**Solution**: Ensure first write completes before second write starts

### 5.4.4 Cross-Pipeline Hazards

Different pipeline domains (memory, vector, matrix) may have different latencies:

```
Memory Pipeline:    Load --> Prefetch --> ...
Vector Pipeline:   Compute --> Compute --> ...
Matrix Pipeline:   MatMul --> MatMul --> ...

Cross-pipeline hazard: Need sync between memory and compute
```

```cpp
Event e_load, e_compute;
TLOAD(tile, gmem, e_load);      // Memory pipeline
TSYNC(e_load);                    // Sync before compute
TMATMUL(acc, tile, weight);      // Matrix pipeline
```

A backend MAY internally optimize hazard handling, but MUST preserve architecture-observable ordering.

## 5.5 Event and Dependency Model

### 5.5.1 Event Properties

The event model provides:

| Property | Description |
|----------|-------------|
| Creation | Events are created implicitly by operations |
| Association | Operations produce events when they complete |
| Consumption | TSYNC consumes events to establish ordering |
| Scope | Events are local to a core (implementation-defined scope) |

### 5.5.2 Event Lifecycle

```
1. Create Event (implicit)
   Event e;

2. Associate with Operation
   TLOAD(tile, gmem, e);  // Load will produce event e

3. Wait for Event
   TSYNC(e);              // Block until e completes

4. Event State Transitions
   Created --> In-flight --> Completed
```

### 5.5.3 Deterministic Dependencies

The event model MUST provide a deterministic dependency relation suitable for:

- Pipeline handoff between producer and consumer instruction groups
- Safe reuse of tile and memory resources
- Reproducible execution under equivalent program order and dependency specification

```cpp
// Deterministic: explicit event controls ordering
Event e0, e1;
TLOAD(A, gmemA, e0);    // e0 tracks Load A
TLOAD(B, gmemB, e1);    // e1 tracks Load B
TSYNC(e0);               // Wait for A
TSYNC(e1);               // Wait for B
// Both loads complete before compute
```

## 5.6 Auto vs Manual Synchronization Responsibilities

### 5.6.1 Auto Mode

In Auto mode, the compiler/runtime is responsible for:

- Inserting required synchronization for legal execution
- Managing tile placement and reuse
- Ensuring data dependencies are satisfied

**Example:**
```cpp
void kernel_auto() {
    TileT src0, src1, dst;
    
    // Compiler inserts synchronization automatically
    TLOAD(src0, gmem0);   // May add implicit sync
    TLOAD(src1, gmem1);   // May add implicit sync
    
    TADD(dst, src0, src1);  // Compiler ensures operands ready
    
    TSTORE(gmem_out, dst);   // May add implicit sync
}
```

### 5.6.2 Manual Mode

In Manual mode, programmers are responsible for:

- Providing required synchronization when dependencies are not otherwise guaranteed
- Explicitly managing tile lifetime and reuse
- Ensuring memory ordering

**Example:**
```cpp
void kernel_manual() {
    TileT src0, src1, dst;
    
    // Explicit placement
    TASSIGN(src0, 0x1000);
    TASSIGN(src1, 0x2000);
    TASSIGN(dst,  0x3000);
    
    // Explicit synchronization
    Event e0, e1;
    TLOAD(src0, gmem0, e0);
    TLOAD(src1, gmem1, e1);
    
    TSYNC(e0);
    TSYNC(e1);
    
    TADD(dst, src0, src1);
    
    Event e2;
    TSTORE(gmem_out, dst, e2);
    TSYNC(e2);
}
```

### 5.6.3 Toolchain Responsibilities

Toolchains MUST NOT:

- Remove required user-authored synchronization unless a provably equivalent ordering is preserved
- Reorder operations that would violate explicit dependencies
- Optimize away synchronization that affects observable behavior

Toolchains MAY:

- Insert additional synchronization in Auto mode
- Reorder independent operations
- Optimize within synchronization boundaries

## 5.7 Diagnostics Requirements

Synchronization diagnostics SHOULD include:

### 5.7.1 Required Information

| Diagnostic Field | Description |
|-----------------|-------------|
| Missing dependency | Required event not provided |
| Invalid dependency | Event in invalid state |
| Ordering violation | Operations reordered illegally |
| Backend limitation | Synchronization form not supported |

### 5.7.2 Error Classes

| Error Class | Description |
|-------------|-------------|
| `PTO-SYNC-001` | Missing event for dependency |
| `PTO-SYNC-002` | Invalid event state |
| `PTO-SYNC-003` | Ordering constraint violated |
| `PTO-SYNC-004` | Backend capability limitation |

### 5.7.3 Example Diagnostics

```
Error [PTO-SYNC-001]: Missing synchronization for RAW hazard
  Operation: pto.tadd
  Reason: tile0 is written by prior store but read without waiting
  Hint: Add TSYNC between store and load operations

Error [PTO-SYNC-003]: Ordering constraint violated
  Operation: pto.tmatmul
  Reason: Producer operation not guaranteed complete before consumer
  Hint: Use Event + TSYNC to establish ordering
```

## 5.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Event API | docs/coding/Event.md |
| Memory Ordering | Chapter 11 |
| TSYNC Instruction | docs/isa/TSYNC.md |
| Programming Guide | Chapter 8 |
