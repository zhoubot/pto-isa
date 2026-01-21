# PTO ISA Compiler - Updates (January 20, 2026)

This document summarizes all the updates made to the PTO ISA Compiler project today.

## Table of Contents

1. [Tile Shape Fix in Orchestration Code](#1-tile-shape-fix-in-orchestration-code)
2. [Recursive Binary Expansion for Nested Loops](#2-recursive-binary-expansion-for-nested-loops)
3. [CALL Instruction Offset Support](#3-call-instruction-offset-support)
4. [Platform-Independent Orchestration](#4-platform-independent-orchestration)
5. [PTO Runtime System](#5-pto-runtime-system)
6. [LLaMA 7B Flash Attention](#6-llama-7b-flash-attention)
7. [Task Graph Visualization](#7-task-graph-visualization)
8. [Dynamic Tiling Support](#8-dynamic-tiling-support)
9. [Performance Results](#9-performance-results)
10. [SRAM Constraint Analysis](#10-sram-constraint-analysis)
11. [Memory Estimation Clarification](#11-memory-estimation-clarification)

---

## 1. Tile Shape Fix in Orchestration Code

### Problem
The generated orchestration code was using hardcoded tile shapes `(8, 8)` instead of actual tile dimensions `(32, 128)` or `(64, 128)`.

### Solution
- Fixed regex pattern in `transform_func_names` to correctly match 5 arguments before rows/cols
- Transform both function names AND tile shapes when using adaptive tiles
- Added `tile_levels[0]` as special key for residual tile size (32-row)

### Result
```c
// Before (wrong)
pto_task_add_input(rt, t0, input, offset, 0, 8, 8);

// After (correct)
pto_task_add_input(rt, t0, input, offset, 0, 64, 128);  // For 64-row tiles
pto_task_add_input(rt, t0, input, offset, 0, 32, 128);  // For 32-row tiles (residual)
```

---

## 2. Recursive Binary Expansion for Nested Loops

### Problem
Binary expansion was only applied to the outer loop, leaving inner loops (like Flash Attention's KV loop) unoptimized.

### Solution
`apply_binary_expansion()` now recursively processes loop bodies, enabling N² → (N/2)² optimization for nested loops.

### Result
For Flash Attention (256 tiles = 8K sequence):
| Configuration | Outer | Inner | Tasks | Reduction |
|--------------|-------|-------|-------|-----------|
| No Optimization | 256 | 256 | ~200K | - |
| **With Optimization** | 128 | 128 | 51,200 | **75%** |

---

## 3. CALL Instruction Offset Support

### New Syntax
```python
# Simple format (no offset)
.call("func", {"param": "tensor"})

# Offset format (for dynamic tiling)
.call("func", {"param": ("tensor", "row_offset", col_offset)})
```

### Example
```python
# Phase 1: Per-tile parallel processing
.call("rmsnorm_tile", {
    "input": ("input", "tile_i", 0),       # input[tile_i]
    "output": ("temp_norm", "tile_i", 0)
})

# Phase 2: Cross-tile dependencies
.call("flash_attn_score_block", {
    "input_q": ("all_q_rope", "q_tile", 0),   # Q[q_tile]
    "input_k": ("all_k_rope", "kv_tile", 0),  # K[kv_tile] ← Cross-dependency!
    "output_s": ("temp_scores", "q_tile", 0)
})
```

---

## 4. Platform-Independent Orchestration

### Architecture Change
Removed platform-specific orchestration generation (`_generate_cuda_orchestration`, `_generate_ascend_orchestration`).

**Correct Architecture:**
```
Orchestration Function
├── Pure C code (platform-independent)
├── Calls PTO Runtime APIs:
│   ├── pto_task_alloc()
│   ├── pto_task_add_input()
│   ├── pto_task_add_output()
│   └── pto_task_submit()
└── Same .c file used for ARM64/CUDA/Ascend backends
```

Only InCore functions differ per backend (CUDA kernels, Ascend C++, ARM64 NEON).

---

## 5. PTO Runtime System

### New Files
- `pto_runtime.h` - Runtime data structures
- `pto_runtime.c` - Runtime implementation

### Key Data Structures
```c
typedef struct {
    char         func_name[64];       // InCore function name
    int32_t      fanin;               // Dependency count
    int32_t      fanout[512];         // Downstream task IDs
    TaskArg      args[32];            // Input/output arguments
    int32_t      buffer_size;         // InCore buffer size (KB)
} PendingTask;

typedef struct {
    PendingTask  pend_task[262144];   // Task table
    TensorMap    tensor_map;          // Tensor → producer mapping
    int32_t      ready_queue[1024];   // Ready tasks
} PTORuntime;
```

### APIs
- `pto_task_alloc()` - Allocate task ID
- `pto_task_add_input/output()` - Track tensor dependencies
- `pto_task_submit()` - Submit task, build dependency graph
- `pto_runtime_dump()` - Dump task graph to file

---

## 6. LLaMA 7B Flash Attention

### Three-Phase Architecture
```
Phase 1: Pre-Attention (Parallel)
├── RMSNorm
├── Q/K/V Projections
└── RoPE Encoding

Phase 2: Flash Attention (N×N Cross-Dependencies)
├── Score Block Computation
├── Online Softmax Update
└── Output Accumulation

Phase 3: Post-Attention/FFN (Parallel)
├── Output Projection
├── Residual Connection
├── FFN (SwiGLU)
└── Final Residual
```

### Adaptive Tile Sizing
```python
TILE_ROWS_BY_LEVEL = {
    4096: 64,   # 128K seq: use 64-row tiles
    2048: 64,   # 64K seq: use 64-row tiles
    1024: 64,   # 32K seq: use 64-row tiles
    512:  64,   # 16K seq: use 64-row tiles
    256:  64,   # 8K seq: use 64-row tiles
    0:    32,   # Residual: use 32-row tiles
}
```

---

## 7. Task Graph Visualization

### New Tool
`visualize_taskgraph.py` - Convert task dump to PDF

### Usage
```bash
python visualize_taskgraph.py <dump_file> <output_pdf>
```

### Features
- Left-to-right layout (rankdir=LR)
- Same-level tasks in same column (rank=same)
- Shows function name and buffer size
- Arrows indicate dependencies

---

## 8. Dynamic Tiling Support

### New Module
`pto_dynamic_tiling.py` with helpers:
- `compute_tile_shape()` - Calculate optimal tile shape
- `build_unary_op()`, `build_binary_op()` - Build tiled operations

### Tile Sizes by ISA
| ISA | F32 | F16 | F64 |
|-----|-----|-----|-----|
| ARM64 | 1×4096 | 1×8192 | 1×2048 |
| CUDA | 1×4096 | 1×8192 | 1×2048 |
| Ascend 910B | 32×128 | 32×256 | 32×64 |

### Control Flow Support
- `FOR` loop with scalar variable bounds
- `IF/ENDIF` for tail handling
- Scalar operations for loop control

---

## 9. Performance Results

### LLaMA 7B with Adaptive Tile Optimization

| SeqLen | Tiles | ActIter | Tasks | No-Opt | Build | Tasks/ms | Memory | Saved |
|--------|-------|---------|-------|--------|-------|----------|--------|-------|
| 1K | 32 | 16 | 1,024 | 3,584 | 0.14ms | 7,111 | 2.8MB | 71% |
| 2K | 64 | 32 | 3,584 | 13,312 | 0.44ms | 8,072 | 9.8MB | 73% |
| 4K | 128 | 64 | 13,312 | 51,200 | 1.7ms | 7,849 | 36MB | 74% |
| 8K | 256 | 128 | 51,200 | 200,704 | 8.5ms | 6,021 | 140MB | 75% |
| 16K | 512 | 256 | 200,704 | 794,624 | 34ms | 5,868 | 548MB | **75%** |

### 128K Extrapolation

| Metric | 16K | 128K (Est.) | Ratio |
|--------|-----|-------------|-------|
| Tiles | 512 | 4,096 | 8x |
| Tasks | 200,704 | 12,611,584 | 63x |
| Build Time | 34ms | ~2.1s | 61x |
| Memory | 548MB | ~33.6GB | 63x |

### Key Formulas
- **Without optimization**: `Tasks = 16N + 3N²`
- **With optimization**: `Tasks = 7N + 3N²/4`
- **Reduction**: N² → (N/2)² = N²/4 → **75% task reduction**

---

## 10. SRAM Constraint Analysis

### Why Tile Size is Limited to 64 Rows

Flash Attention memory layout per block:
```
Q:  tile_rows × head_dim
K:  tile_rows × head_dim
V:  tile_rows × head_dim
S:  tile_rows × tile_rows  ← QUADRATIC!
O:  tile_rows × head_dim
```

### SRAM Usage by Tile Size

| Tile Rows | Q/K/V | S (Score) | Total | Fits 256KB? |
|-----------|-------|-----------|-------|-------------|
| 32 | 16KB each | 4KB | 68KB | ✓ |
| **64** | 32KB each | **16KB** | **144KB** | ✓ |
| 128 | 64KB each | **64KB** | 321KB | ✗ |
| 256 | 128KB each | **256KB** | 770KB | ✗ |

**Score matrix S = tile_rows² is the bottleneck!**

### Implication
With 256KB SRAM (Ascend 910B), maximum tile size is 64 rows. To use larger tiles (128+), would need 512KB+ SRAM.

---

## 11. Memory Estimation Clarification

### What IS Included (Task Management Only)

| Data Structure | Size | Purpose |
|----------------|------|---------|
| **PendingTask** | 2,864 bytes/task | Task metadata + fanout[512] array |
| **TensorMapEntry** | 56 bytes/entry | Dependency tracking hash table |
| **Ready Queue** | 16 KB (fixed) | Ready task queue |

The largest component of `PendingTask` is the **fanout[512]** array (2 KB), which stores downstream dependent task IDs.

### What is NOT Included

| Memory Type | LLaMA-7B Scale |
|-------------|----------------|
| **Model Weights** (W_q, W_k, W_v, W_o, W_gate, W_up, W_down) | ~14 GB (fp16) / ~7 GB (int8) |
| **Input/Output Tensors** | Depends on batch size and seq_len |
| **InCore Tile Buffers** (Q, K, V, S, O per block) | ~144 KB per block |
| **KV Cache** | ~4 GB for 16K context |
| **Intermediate Activations** | Proportional to seq_len |

### Example Comparison (SeqLen = 16K)

```
Task Management Memory:    ~548 MB    (what we estimate)
Model Weights:            ~14,000 MB  (fp16)
KV Cache:                  ~4,000 MB  (estimated)
─────────────────────────────────────
Total (actual inference): ~18,500 MB
```

**Conclusion**: Our memory estimates reflect only the overhead of building the task graph in the Orchestration function, NOT the memory required to actually execute the InCore functions with real model weights and activations.

---

## Summary

Today's updates significantly improved the PTO compiler with:

1. **Correctness**: Fixed tile shape generation in orchestration code
2. **Optimization**: 75% task reduction via adaptive tiles and recursive binary expansion
3. **Architecture**: Clean separation of platform-independent orchestration
4. **Tooling**: Runtime system, task dump, and visualization
5. **Analysis**: SRAM constraint documentation and 128K extrapolation

The compiler now correctly handles LLaMA 7B with Flash Attention for sequence lengths up to 16K, with clear analysis of limitations for longer sequences.
