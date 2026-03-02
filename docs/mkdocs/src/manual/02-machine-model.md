# 2. Machine Model

This chapter defines the abstract execution model that Virtual ISA programs target. It specifies architecture-visible ordering and responsibility boundaries, not microarchitecture internals.

## 2.1 Scope

The PTO machine model defines:

- **Execution agents**: Host, Device, and Core machines
- **Program granularity**: Tile-level operation sequences
- **Ordering domains**: Program order, event/synchronization, and memory visibility
- **Responsibility models**: Auto and Manual programming modes

This model provides a stable abstraction that enables software portability across different PTO implementations while allowing flexibility in internal architecture.

## 2.2 Execution Agents

The abstract PTO machine has three conceptual agents, each with distinct responsibilities:

### 2.2.1 Host Machine

The host machine is responsible for:

- Preparing workloads and kernel arguments
- Submitting execution to the device
- Managing global resources and memory allocation
- Coordinating overall application flow

```
Host Application
    |
    v
Submit Kernel(...)
    |
    v
Device Queue
```

### 2.2.2 Device Machine

The device machine is responsible for:

- Scheduling tile programs across execution resources
- Managing work distribution to available cores
- Coordinating data movement between global memory and tiles
- Handling runtime synchronization

```
Device Machine
    |
    +-- Work Scheduler --> Core 0
    +-- Work Scheduler --> Core 1
    +-- Work Scheduler --> ... 
    +-- Memory Manager --> Global Memory
```

### 2.2.3 Core Machine

The core machine is responsible for:

- Executing tile and scalar instructions
- Performing synchronization primitives
- Managing on-chip tile storage (Unified Buffer)
- Processing vector, matrix, and scalar operations

```
Core Machine
    |
    +-- Execution Unit (Vector/Matrix/Scalar)
    +-- Unified Buffer (Tile Storage)
    +-- Synchronization Unit
    +-- Load/Store Unit
```

### 2.2.4 Agent Mapping

A conforming implementation MAY map these agents differently internally, but MUST preserve architecture-visible behavior. For example:

- Single-chip devices may combine Device and Core functionality
- Multi-chip systems may distribute work across multiple Device machines
- Simulation environments may map Core operations to host CPU

## 2.3 Program Granularity

PTO programs operate at tile granularity, meaning the fundamental unit of work is a tile operation.

### 2.3.1 Program Structure

A PTO program is an ordered sequence of operations over:

- **Tile values**: 2D arrays of elements
- **Scalar values**: Immediate values and constants
- **Memory references**: Global memory views
- **Event values**: Synchronization tokens

### 2.3.2 Concurrent Execution

Execution units MAY process independent tile programs concurrently. The architecture defines:

- Independent operations MAY execute in any order or concurrently
- Dependent operations MUST observe required happens-before relations
- The program order defines the baseline ordering

### 2.3.3 Example: Program Structure

```cpp
// PTO program structure
void kernel_example() {
    // Independent loads - may execute concurrently
    TLOAD(tileA, gmemA);  // Load 1
    TLOAD(tileB, gmemB);  // Load 2 - independent of Load 1
    
    // Synchronization point
    TSYNC();  // Wait for all prior operations
    
    // Dependent compute - must happen after loads
    TMATMUL(tileC, tileA, tileB);
    
    // Store - must happen after compute
    TSTORE(gmemC, tileC);
}
```

## 2.4 Dispatch and Scheduling

### 2.4.1 Scheduling Policy

Scheduling policy is implementation-defined, subject to architecture rules:

- Independent work MAY execute out of order
- Dependence-ordered work MUST observe required happens-before relations
- Backend/runtime MAY use SPMD, MPMD, or hybrid dispatch models

### 2.4.2 SPMD Model (Single Program, Multiple Data)

In SPMD, all cores execute the same program but operate on different data:

```
Core 0: kernel(entry=0) --> processes tile block 0
Core 1: kernel(entry=0) --> processes tile block 1
Core N: kernel(entry=0) --> processes tile block N
```

Example:
```cpp
__global__ void kernel_spmd(int block_id) {
    // Each block processes different data based on block_id
    int tile_idx = block_id;
    process_tile(tile_idx);
}
```

### 2.4.3 MPMD Model (Multiple Program, Multiple Data)

In MPMD, different cores may execute different programs or different control paths:

```
Core 0: program_A --> produces data
Core 1: program_A --> produces data
Core N: program_B --> consumes data
```

Example:
```cpp
void kernel_mpmd(uint32_t task_id) {
    switch (task_id) {
        case 0: return producer_stage();
        case 1: return consumer_stage();
    }
}
```

### 2.4.4 Hybrid Model

PTO supports hybrid models where:

- Multiple programs coexist in a single kernel
- Different cores may have different roles
- Task distribution is runtime-determined

## 2.5 Architecture-Visible Ordering Domains

Ordering is defined across three distinct domains:

### 2.5.1 Program Order Domain

Within a single dependent chain, later operations MUST observe earlier committed effects.

```
Program Order:
    Operation A (before)
         |
         v
    Operation B (after)
         |
         v
    Operation C (after)
```

Example:
```cpp
TLOAD(tileA, gmemA);   // Operation A
TMATMUL(tileC, tileA, tileB);  // Operation B depends on A
TSTORE(gmemC, tileC);  // Operation C depends on B
```

### 2.5.2 Event/Synchronization Domain

Event operations and `TSYNC` establish architecture-defined ordering points:

```
Without synchronization:
    Core 0: Load A --> Compute A --> Store A  (may be out of order)

With TSYNC:
    Core 0: Load A --> Compute A --> Store A 
         |              |            |
         v              v            v
    TSYNC(e)-------->TSYNC(e)---->TSYNC(e)
         |
         +--> Core 1 can now see results
```

### 2.5.3 Memory Visibility Domain

`TLOAD`/`TSTORE` visibility rules apply according to memory-ordering constraints:

- Load operations see data from prior stores (with synchronization)
- Memory model defines visibility across cores
- See Chapter 11 for detailed memory ordering semantics

## 2.6 Auto vs Manual Responsibilities

PTO supports two architecture-level responsibility modes that define how computational resources are managed:

### 2.6.1 Auto Mode

In Auto mode, the compiler/runtime manages:

| Responsibility | Managed By |
|---------------|------------|
| Tile placement | Compiler/Runtime |
| Address frontend | Compiler/Runtime |
| Synchronization | Compiler/Runtime |
| Operation scheduling | Compiler/Runtime |

**Characteristics:**
- Compiler inserts required synchronization automatically
- Placement decisions are tool-managed
- User intent remains architecture-visible but operational details are tool-managed
- Best for productivity and portability

**Example:**
```cpp
void kernel_auto() {
    TileT src0, src1, dst;
    
    // Compiler handles:
    // - Where tiles are placed in UB
    // - When to synchronize
    // - How to schedule operations
    
    TLOAD(src0, gmem0);
    TLOAD(src1, gmem1);
    TADD(dst, src0, src1);  // Compiler inserts sync if needed
    TSTORE(gmem_out, dst);
}
```

### 2.6.2 Manual Mode

In Manual mode, the programmer is responsible for:

| Responsibility | Managed By |
|---------------|------------|
| Tile placement | Programmer (via TASSIGN) |
| Address frontend | Programmer |
| Synchronization | Programmer (via Events/TSYNC) |
| Operation scheduling | Programmer |

**Characteristics:**
- Programmer controls memory placement explicitly
- Programmer expresses ordering through events
- Toolchain MUST preserve explicitly authored synchronization
- Best for performance-critical kernels

**Example:**
```cpp
void kernel_manual() {
    TileT src0, src1, dst;
    
    // Programmer controls everything:
    TASSIGN(src0, 0x1000);  // Explicit address
    TASSIGN(src1, 0x2000);
    TASSIGN(dst,  0x3000);
    
    Event e0, e1;
    TLOAD(src0, gmem0, e0);
    TLOAD(src1, gmem1, e1);
    TSYNC(e0);
    TSYNC(e1);
    
    TADD(dst, src0, src1);  // Programmer ensures sync
    
    Event e2;
    TSTORE(gmem_out, dst, e2);
    TSYNC(e2);
}
```

### 2.6.3 Mixed Mode

Both modes can be combined within a single program:
```cpp
void kernel_mixed() {
    // Manual: critical tiles
    TileT acc;
    TASSIGN(acc, 0x8000);
    
    // Auto: helper tiles
    TileT tmp0, tmp1;
    TLOAD(tmp0, gmem0);  // Auto placement
    
    // Compute
    TMATMUL(acc, tmp0, tmp1);  // Uses manually-placed acc
}
```

## 2.7 Implementation-Defined Surface

The following remain implementation-defined and MUST be documented per backend profile:

### 2.7.1 Scheduler Heuristics

- How tiles are assigned to cores
- Instruction-level parallelism decisions
- Pipeline utilization strategies

### 2.7.2 Pipeline Details

- Pipeline occupancy and issue rates
- Latency hiding strategies
- Vector/matrix pipeline coordination

### 2.7.3 Buffering

- Internal buffering and transient placement
- Cache management policies
- Prefetch strategies

### 2.7.4 Backend-Specific Constraints

- Supported tile shapes per operation
- Legal dtype combinations
- Location-intent restrictions

> **Important**: These details MUST NOT change architecture-defined instruction semantics. The observable behavior of a correctly-synchronized program must be consistent across implementations.

## 2.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Synchronization | Chapter 5 |
| Memory Ordering | Chapter 11 |
| Backend Profiles | Chapter 12 |
| Programming Guide | Chapter 8 |
