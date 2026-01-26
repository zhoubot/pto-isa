<p align="center">
  <img src="assets/pto_logo.svg" alt="PTO logo" width="180" />
</p>

# PTO ISA Compiler

Parallel Tile Operations (PTO) — a small tile-ISA + compiler toolchain for heterogeneous accelerators.

## PTO-AS (`ptoas`) Quickstart

This repo includes the **PTO-AS toolchain**:

- `bin/ptoas`: wrapper that dispatches to an OS/arch-specific `ptoas` binary
- `ptoas/`: PTO-AS sources + Python frontend (`ptoas/python/`) + tools (`ptoas/tools/`)
- `kernels/python/`: Python → PTO-AS → `ptoas` → run/compare (CPU ref; optional NPU)

### 0) `ptoas` binary

- Linux aarch64: `bin/linux-aarch64/ptoas` is included.
- macOS aarch64: build `ptoas` and place it at `bin/macos-aarch64/ptoas` (see `bin/README.md` and `ptoas/mlir/README.md`).

### 1) CPU-only (Ubuntu/macOS aarch64)

```bash
python3 ptoas/tools/run_e2e_cpu.py --ptoas ./bin/ptoas --outdir /tmp/ptoas_e2e_cpu
```

### 2) Ascend NPU (Ubuntu aarch64)

```bash
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest

python3 ptoas/tools/run_e2e_npu.py \
  --ptoas ./bin/ptoas \
  --ascend-home $ASCEND_HOME_PATH \
  --run-mode npu \
  --device 0 \
  --block-dim 1 \
  --outdir /tmp/ptoas_e2e_npu
```

### 3) Regression subset

```bash
python3 kernels/python/run_regression.py \
  --ptoas ./bin/ptoas \
  --ascend-home $ASCEND_HOME_PATH \
  --run-mode npu \
  --device 0 \
  --block-dim 1 \
  --cases add16,gemm16,rowmax16,softmax16
```

For simulator runs, use `--run-mode sim --soc a3`.

## Old PTOFunctionBuilder → PTO-AS (GEMM16)

The original frontend lives under `src/compile` (`PTOFunctionBuilder`). For a small end-to-end GEMM example, export it to the **new-format PTO-AS** and compile/run it via `ptoas`:

```bash
# NPU (Ubuntu aarch64 + Ascend)
export ASCEND_HOME_PATH=$HOME/Ascend/ascend-toolkit/latest
python3 scripts/ptoas/regenerate_gemm16_pto_test.py --run both --ptoas ./bin/ptoas --device 0

# CPU-only (Ubuntu/macOS aarch64)
python3 scripts/ptoas/regenerate_gemm16_pto_test.py --run cpu --ptoas ./bin/ptoas
```

Outputs are written under `build/pto_from_pto/`:
- `build/pto_from_pto/output_pto/gemm16_pto_test/gemm16.pto`
- `build/pto_from_pto/output_ascend_a2a3/gemm16_pto_test/gemm16.cpp` (+ `gemm16.bin`)

## Overview

PTO ISA Compiler is a comprehensive framework for developing high-performance tile-based computations on heterogeneous hardware platforms. It provides:

- **Python DSL** for defining tile operations and orchestration logic
- **Multi-backend code generation** (ARM64 NEON, CUDA, Ascend NPU)
- **Task-parallel runtime** with dependency tracking and dynamic scheduling
- **Cycle-accurate simulator** for Ascend A2/A3 NPU architecture
- **Performance benchmarking** and profiling tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PTO ISA Compiler                            │
├─────────────────────────────────────────────────────────────────────┤
│  Python DSL (PTOFunctionBuilder)                                    │
│    ├── InCore Functions (tile-level operations)                     │
│    └── Orchestration Functions (task scheduling)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Code Generators                                                    │
│    ├── pto_codegen_arm64.py      → ARM64 NEON C code               │
│    ├── pto_codegen_cuda.py       → CUDA kernels                    │
│    ├── pto_codegen_ascend.py     → Ascend AscendC code             │
│    └── pto_codegen_ascend_a2a3_sim.py → Cycle-accurate simulator   │
├─────────────────────────────────────────────────────────────────────┤
│  Runtime System (C)                                                 │
│    ├── Task Management (sliding window, 8K tasks in-flight)        │
│    ├── TensorMap (dependency tracking via hash table)              │
│    ├── Dual Ready Queues (Vector + Cube for A2A3)                  │
│    └── Cycle Tracing (Chrome Tracing format)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Features

### 1. Two-Level Programming Model

**InCore Functions** - Tile-level computations that fit within on-chip SRAM:
```python
# Define a tile operation
func = (PTOFunctionBuilder("tile_add")
    .in_core()
    .tile("a", 32, 128, ElementType.F32)
    .tile("b", 32, 128, ElementType.F32)
    .tile("c", 32, 128, ElementType.F32)
    .memref("input_a", MemorySpace.GM, ElementType.F32)
    .memref("input_b", MemorySpace.GM, ElementType.F32)
    .memref("output", MemorySpace.GM, ElementType.F32)
    .load("a", "input_a", 0, 0)
    .load("b", "input_b", 0, 0)
    .add("c", "a", "b")
    .store("c", "output", 0, 0)
    .build())
```

**Orchestration Functions** - Dynamic scheduling across tiles:
```python
# Define orchestration with dynamic loops
orch = (PTOFunctionBuilder("process_sequence")
    .orchestration()
    .for_loop("tile_i", 0, "num_tiles", 1)
        .call("tile_matmul", {
            "input_a": ("data", "tile_i", 0),
            "input_b": "weights",
            "output": ("output", "tile_i", 0)
        })
    .end_for()
    .build())
```

### 2. Automatic Dependency Tracking

The runtime automatically tracks data dependencies between tasks using a TensorMap:
- Producer tasks register their output regions
- Consumer tasks query for dependencies
- The scheduler ensures correct execution order

### 3. Dual-Queue Architecture (Ascend A2/A3)

For Ascend NPUs, tasks are routed to specialized execution units:
- **Vector Workers (48)**: Element-wise ops, reductions, activations
- **Cube Workers (24)**: Matrix multiplication (GEMM/MATMUL)

### 4. Binary Expansion for Dynamic Loops

Loops with runtime-variable bounds are expanded into power-of-2 blocks for better performance:
```c
// Original: for (i = 0; i < num_tiles; i++)
// Expanded:
if (rem >= 256) { for (i = 0; i < 256; i++) {...} rem -= 256; }
if (rem >= 128) { for (i = 0; i < 128; i++) {...} rem -= 128; }
// ... handles any num_tiles value efficiently
```

## Project Structure

```
PTO_ISA_Compiler/
├── src/
│   ├── isa_definition/
│   │   └── pto_isa_definition.py    # Complete ISA specification
│   ├── compile/
│   │   ├── pto_compile.py           # PTOFunctionBuilder DSL
│   │   ├── pto_compile_common.py    # Common compiler infrastructure
│   │   ├── pto_codegen_arm64.py     # ARM64 NEON backend
│   │   ├── pto_codegen_cuda.py      # CUDA backend
│   │   ├── pto_codegen_ascend.py    # Ascend NPU backend
│   │   └── pto_codegen_ascend_a2a3_sim.py  # Cycle simulator codegen
│   └── runtime/
│       ├── pto_runtime_common.h/c   # Platform-independent runtime
│       ├── pto_runtime_arm64.h/c    # ARM64 worker threads
│       ├── pto_runtime_a2a3.h/c     # Ascend dual-queue runtime
│       └── ascend_a2a3_core_model/  # A2A3 Core Simulator (C library)
│           ├── a2a3_core_model.h/c  # Core model (pipes, sync)
│           ├── a2a3_incore_sim.h/c  # InCore function simulator
│           └── Makefile             # Build system
├── scripts/
│   ├── visualize_taskgraph.py       # Task graph visualization tool
│   ├── run_examples.py              # Batch example runner
│   ├── a3/                          # Ascend A3 specific scripts
│   └── cpu/                         # CPU runner utilities
├── examples/
│   ├── llama/                       # LLaMA 7B decoder layer
│   └── softmax/                     # Fused softmax example
├── config_example.py                # Configuration tool
└── docs/
    └── *.md                         # Documentation
```

## Tools

### Task Graph Visualizer

Visualize task dependency graphs from runtime dump files:

```bash
# Generate PDF from task dump
python scripts/visualize_taskgraph.py output/ascend_a2a3_sim/llama_layer_dynamic_task_graph.txt

# Generate PNG format
python scripts/visualize_taskgraph.py task_dump.txt output.png --format png

# Generate DOT file only (for custom rendering)
python scripts/visualize_taskgraph.py task_dump.txt --dot-only output.dot
```

Features:
- Parses simple and verbose task dump formats
- Computes task levels for parallel visualization
- Color-codes tasks by function type
- Shows buffer sizes and dependency counts

## Supported Operations

### Tile Operations (InCore)
| Category | Operations |
|----------|------------|
| Arithmetic | `add`, `sub`, `mul`, `div`, `neg`, `abs` |
| Math | `exp`, `log`, `sqrt`, `rsqrt`, `pow` |
| Activation | `relu`, `sigmoid`, `tanh`, `silu`, `gelu` |
| Comparison | `max`, `min`, `cmp` |
| Reduction | `rowsum`, `rowmax`, `rowmin`, `colsum`, `colmax` |
| Broadcast | `rowexpandmul`, `rowexpanddiv`, `rowexpandsub` |
| Matrix | `matmul` (Cube unit on A2A3) |
| Memory | `load`, `store`, `copy` |

### Scalar Operations
`sadd`, `ssub`, `smul`, `sdiv`, `smov`, `sli`, `scmp`

### Control Flow
`for_loop`, `while_loop`, `if/else`, `call`, `return`

## Quick Start

### Step 1: Configure with `config_example.py`

The configuration tool provides an interactive menu to set up your build:

```bash
python config_example.py
```

**Menu Options:**

```
=== PTO Example Configuration ===

Current Configuration:
  Example: examples/llama
  Platform: ascend_a2a3_sim (Ascend A2/A3 Cycle Simulator)
  Binary Expansion: ON
  Task Dump: ON
  Task Graph PDF: ON
  Orchestration Benchmark: ON
  Runtime Benchmark: ON
  Sequence Length Range: 1024-16384 (step 1024)
  Accuracy Test: ON
  Simulation: ON
  Trace Generation: ON

Options:
  1. Select Example Directory
  2. Select Target Platform
  3. Toggle Binary Expansion
  4. Toggle Task Dump
  5. Toggle Task Graph PDF
  6. Toggle Orchestration Benchmark
  7. Toggle Runtime Benchmark
  8. Configure Sequence Length Range
  9. Toggle Accuracy Test
  10. Toggle Simulation & Trace
  11. Generate run_<platform>.py
  0. Exit

Enter choice:
```

**Platform Options:**
| Platform | Description | Output |
|----------|-------------|--------|
| `arm64` | ARM64 NEON (CPU) | Native execution |
| `cuda` | NVIDIA CUDA (GPU) | CUDA kernels |
| `ascend_a2a3` | Ascend A2/A3 NPU | AscendC code |
| `ascend_a2a3_sim` | Ascend Cycle Simulator | Cycle-accurate simulation |

After configuration, select option `11` to generate `run_<platform>.py` in the example directory.

### Step 2: Run with `run_xxx.py`

The generated script handles the complete workflow:

```bash
cd examples/llama
python run_ascend_a2a3_sim.py
```

**What `run_xxx.py` Does:**

1. **Code Generation** - Converts PTO Python definitions to C code
   ```
   Generating code for platform: ascend_a2a3_sim
   Output: output/ascend_a2a3_sim/generated_code/
   ```

2. **Compilation** - Compiles generated C code with appropriate toolchain
   ```
   Compiling: gcc -O3 -march=native ...
   Output: output/ascend_a2a3_sim/llama_layer_dynamic
   ```

3. **Task Dump** (if enabled) - Generates task table for analysis
   ```
   Output: output/ascend_a2a3_sim/llama_layer_dynamic_task_graph.txt
   ```

4. **Task Graph PDF** (if enabled) - Visualizes dependency graph
   ```
   Output: output/ascend_a2a3_sim/llama_layer_dynamic_task_graph.pdf
   ```

5. **Orchestration Benchmark** (if enabled) - Measures task submission throughput
   ```
   === Orchestration Benchmark ===
   seq_len   num_tiles   tasks    time_ms    tasks/ms
   1024      32          171      0.020      8550000
   2048      64          339      0.045      7533333
   ...
   ```

6. **Runtime Benchmark** (if enabled) - Measures actual execution/simulation
   ```
   === Runtime Benchmark ===
   seq_len   num_tiles   tasks    time_ms    tasks/ms
   1024      32          171      1.234      138574
   ...
   ```

7. **Simulation & Trace** (if enabled) - Cycle-accurate simulation with trace output
   ```
   [PTO Simulator] Starting Cycle-Accurate Simulation
   Workers: 72 (48 vector + 24 cube)
   Output: output/ascend_a2a3_sim/trace.json
   ```

**Command-Line Options for `run_xxx.py`:**

```bash
# Run with default configuration
python run_ascend_a2a3_sim.py

# Skip code generation (use existing)
python run_ascend_a2a3_sim.py --skip-codegen

# Skip compilation (use existing binary)
python run_ascend_a2a3_sim.py --skip-compile

# Run benchmark only (skip simulation)
python run_ascend_a2a3_sim.py --benchmark-only
```

### Step 3: View Results

**Task Graph (PDF):**
```bash
open output/ascend_a2a3_sim/llama_layer_dynamic_task_graph.pdf
```
Shows task dependencies with parallel tasks vertically aligned.

**Cycle Trace (Chrome Tracing):**
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load" and select `output/ascend_a2a3_sim/trace.json`
4. View timeline with:
   - Vector Workers (pid=0): Element-wise ops, reductions
   - Cube Workers (pid=1): Matrix multiplications

**Benchmark Results (JSON):**
```bash
cat output/ascend_a2a3_sim/benchmark_results.json
```

### Output Directory Structure

```
examples/llama/
├── pto_llama7B_dynamic.py      # PTO module definition
├── run_ascend_a2a3_sim.py      # Generated run script
├── config.json                  # Saved configuration
└── output/
    └── ascend_a2a3_sim/
        ├── generated_code/
        │   ├── llama_layer_dynamic.c    # Main orchestration
        │   ├── rmsnorm_tile.c           # InCore functions
        │   ├── tile_matmul.c
        │   └── ...
        ├── llama_layer_dynamic          # Compiled executable
        ├── llama_layer_dynamic_task_graph.txt   # Task dump
        ├── llama_layer_dynamic_task_graph.pdf   # Visual graph
        ├── trace.json                   # Chrome trace
        └── benchmark_results.json       # Performance data
```

## Example: LLaMA 7B Layer

The `examples/llama/pto_llama7B_dynamic.py` implements a full LLaMA decoder layer:

```
LLaMA 7B Layer Pipeline:
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Pre-Attention (parallel across tiles)              │
│   RMSNorm → Q/K/V Projections (MatMul) → RoPE               │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: Flash Attention (cross-tile dependencies)          │
│   Score blocks → Softmax → Output accumulation              │
├─────────────────────────────────────────────────────────────┤
│ Phase 3: Post-Attention + MLP (parallel)                    │
│   Output Proj → Residual → RMSNorm → SwiGLU → Down Proj    │
└─────────────────────────────────────────────────────────────┘
```

Task distribution for sequence length 16K:
- ~5700 total tasks
- Vector tasks: RMSNorm, RoPE, Softmax, SwiGLU, element-wise ops
- Cube tasks: Q/K/V projections, attention scores, MLP projections

## Performance Benchmarks

### Orchestration Throughput
Measures task submission rate (no actual execution):
```
seq_len=1024:  tasks=171,  throughput=8.55M tasks/ms
seq_len=8192:  tasks=1395, throughput=3.77M tasks/ms
seq_len=16384: tasks=5763, throughput=3.21M tasks/ms
```

### Simulation Statistics
```
Total tasks: 5763
Workers: 72 (48 vector + 24 cube)
Makespan: 12,450 cycles
Active workers: 72/72
```

## Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `target_platform` | arm64, cuda, ascend_a2a3_sim | arm64 |
| `enable_binary_expansion` | Power-of-2 loop unrolling | true |
| `benchmark_orchestration` | Measure task/ms throughput | true |
| `benchmark_runtime` | Measure execution time | true |
| `test_seq_len_min/max/step` | Input range for benchmarks | 1024-16384, step 1024 |
| `enable_simulation` | Cycle-accurate simulation | true |
| `enable_trace_generation` | Chrome tracing output | true |

## Runtime Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `PTO_TASK_WINDOW_SIZE` | 8192 | Sliding window for in-flight tasks |
| `PTO_TENSORMAP_SIZE` | 8192 | Hash table buckets for dependencies |
| `PTO_MAX_WORKERS` | 128 | Maximum worker threads |
| `A2A3_VECTOR_WORKERS` | 48 | Vector unit workers |
| `A2A3_CUBE_WORKERS` | 24 | Cube unit workers |

## Requirements

- Python 3.8+
- GCC (for ARM64 backend)
- NVCC (for CUDA backend, optional)
- Ascend toolchain (for NPU backend, optional)

## License

[Specify your license here]

## References

- [FlexAttention Algorithm Analysis](docs/FlexAttention_Algorithm_Analysis.md)
- [Running on Ascend A3](docs/run_on_ascend_a3.md)
- [PTO ISA Cheatsheet](pto-isa-cheatsheet.pdf)
