# PTO ISA Compiler - Update Log

## 2026-01-23: A2A3 Core Simulator & Dual-Queue Enhancement

### New: Ascend A2/A3 Core Model Simulator

Created a cycle-accurate core model for simulating InCore function execution on Ascend NPU cores.

**Location:** `src/runtime/ascend_a2a3_core_model/`

**Architecture:**

```
CUBE CORE                              VECTOR CORE
┌────────────────────────┐            ┌────────────────────────┐
│  Scalar | MTE Pipes    │            │  Scalar | MTE Pipes    │
│         | GM↔L1, L0C   │            │         | GM↔UB        │
│  ───────┴──────────    │            │  ───────┴──────────    │
│       CUBE Unit        │            │      Vector Unit       │
│    (Matrix Multiply)   │            │  (Elem-wise, Reduce)   │
└────────────────────────┘            └────────────────────────┘
```

**Features:**
- Parallel pipe execution model (Scalar, MTE, Compute)
- Synchronization primitives: SET_FLAG, WAIT_FLAG, PIPE_BARRIER
- Instruction parsing and cycle estimation
- InCore function registration and cached simulation
- Heuristic cycle cost API for runtime integration

**Files:**
| File | Description |
|------|-------------|
| `a2a3_core_model.h/c` | Core model with pipes and sync |
| `a2a3_incore_sim.h/c` | InCore function simulator |
| `test_core_sim.c` | Test program |
| `Makefile` | Build system |

**Test Results:**
```
rmsnorm_tile (11 instructions): 144 cycles
tile_matmul (12 instructions): 276 cycles
```

### Runtime Integration

The core simulator is integrated with the PTO runtime for cycle-accurate simulation:

**Integration Header:** `src/runtime/ascend_a2a3_core/a2a3_sim_integration.h`

**Usage:**
```c
// Compile with core simulator support:
// -DA2A3_CORE_SIM_AVAILABLE -Isrc/runtime -Lsrc/runtime/ascend_a2a3_core_model -la2a3_core

// Automatic integration via pto_estimate_cycle_cost():
int64_t cycles = pto_estimate_cycle_cost("rmsnorm_tile");  // Uses core sim if available

// Manual function registration:
a2a3_sim_register_function("custom_func", false, instruction_code, 32, 128);
```

**Codegen Enhancement:**
- `pto_codegen_ascend_a2a3_sim.py` now generates:
  - Actual Ascend instructions for InCore functions (same as `ascend_a2a3`)
  - Instruction code as string constants for core simulator parsing
  - Registration functions for each InCore function
  - Orchestration code with task submission

**Files Added:**
| File | Description |
|------|-------------|
| `a2a3_sim_integration.h` | Runtime integration header |
| Updated `pto_runtime_common.c` | `pto_estimate_cycle_cost()` uses core sim |
| Updated `pto_codegen_ascend_a2a3_sim.py` | Generates Ascend instructions |

---

## 2026-01-23: Dual-Queue Simulation & Trace Enhancement

### Major Features

#### 1. Correct Vector/Cube Worker Task Routing

**Problem**: Tasks were being assigned to any worker regardless of type, causing vector operations (e.g., `rmsnorm_tile`) to run on cube workers and vice versa.

**Solution**: Modified `pto_simulate_all()` in `pto_runtime_common.c` to respect the `is_cube` flag:

```c
if (rt->dual_queue_mode) {
    if (task->is_cube) {
        // Cube task → assign to cube workers (48-71)
        worker_start = rt->num_vector_workers;
        worker_end = NUM_WORKERS;
    } else {
        // Vector task → assign to vector workers (0-47)
        worker_start = 0;
        worker_end = rt->num_vector_workers;
    }
    // Find worker with lowest cycle count within range
}
```

**Result**:
- `rmsnorm_tile`, `rope_tile`, `softmax_tile`, etc. → Vector Workers (0-47)
- `tile_matmul` → Cube Workers (48-71)

#### 2. Enhanced Chrome Tracing with Worker Type Distinction

**Problem**: Trace file showed 72 workers (tid 0-71) without distinguishing vector vs cube.

**Solution**: Updated `pto_trace_to_chrome_json()` to:
- Use `pid=0` for Vector Workers, `pid=1` for Cube Workers
- Add thread name metadata (`Vector-0`, `Cube-0`, etc.)
- Add process name metadata (`Vector Workers (48)`, `Cube Workers (24)`)

**New trace format**:
```json
{
  "traceEvents": [
    {"name": "thread_name", "ph": "M", "pid": 0, "tid": 0, "args": {"name": "Vector-0"}},
    {"name": "thread_name", "ph": "M", "pid": 1, "tid": 0, "args": {"name": "Cube-0"}},
    {"name": "process_name", "ph": "M", "pid": 0, "args": {"name": "Vector Workers (48)"}},
    {"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "Cube Workers (24)"}},
    {"name": "rmsnorm_tile", "cat": "task", "ph": "X", "ts": 0, "dur": 30, "pid": 0, "tid": 0},
    {"name": "tile_matmul", "cat": "task", "ph": "X", "ts": 30, "dur": 50, "pid": 1, "tid": 0},
    ...
  ]
}
```

#### 3. Auto-Detection of Cube Operations

**Problem**: InCore functions using `.matmul()` weren't being marked as cube operations.

**Solution**: Modified `_make_matmul_method()` in `pto_compile.py` to automatically set `is_cube = True`:

```python
def _make_matmul_method(instr_class, doc):
    def method(self, dst: str, a: str, b: str) -> "PTOFunctionBuilder":
        # Matmul requires cube unit - auto-set is_cube flag
        self.program.is_cube = True
        self._add_instr(instr_class(...))
        return self
    return method
```

#### 4. Fixed `pto_task_alloc` Parameter Order

**Problem**: In A2A3 sim codegen, `is_cube` flag was in wrong parameter position.

**Before** (incorrect):
```c
pto_task_alloc(rt, "func_name", NULL, is_cube, 0, 0);  // is_cube in buffer_bytes position!
```

**After** (correct):
```c
pto_task_alloc(rt, "func_name", NULL, 0, 0, is_cube);  // is_cube in correct position
```

### Files Modified

| File | Changes |
|------|---------|
| `src/runtime/pto_runtime_common.c` | Dual-queue task routing in simulation, enhanced trace export |
| `src/runtime/pto_runtime_common.h` | Added `num_vector_workers`/`num_cube_workers` to CycleTrace |
| `src/runtime/pto_runtime_a2a3.c` | Updated to use `pto_trace_init_dual()` |
| `src/compile/pto_compile.py` | Auto-set `is_cube` for matmul operations |
| `src/compile/pto_codegen_ascend_a2a3_sim.py` | Fixed `pto_task_alloc` parameter order |

### Files Reorganized

| File | Change |
|------|--------|
| `visualize_taskgraph.py` | Moved from root to `scripts/visualize_taskgraph.py` |

### Documentation Added

| File | Description |
|------|-------------|
| `README.md` | Complete project documentation |
| `update.md` | Update log with today's changes |

### New API

```c
// Initialize trace with vector/cube worker distinction
void pto_trace_init_dual(int32_t num_vector_workers, int32_t num_cube_workers);
```

### Verification

Run the LLaMA example and check trace.json:
```bash
cd examples/llama
python run_ascend_a2a3_sim.py
```

Open `output/ascend_a2a3_sim/trace.json` in Chrome's `chrome://tracing`:
- Vector operations should appear in "Vector Workers" process group
- Matmul operations should appear in "Cube Workers" process group

### Task Routing Summary

| Function | Contains Matmul | `is_cube` | Worker Type |
|----------|----------------|-----------|-------------|
| `rmsnorm_tile` | No | false | Vector (0-47) |
| `tile_matmul` | Yes | true | Cube (48-71) |
| `rope_tile` | No | false | Vector (0-47) |
| `softmax_tile` | No | false | Vector (0-47) |
| `swiglu_tile` | No | false | Vector (0-47) |
| `attention_score_tile` | Yes | true | Cube (48-71) |
| `attention_output_tile` | Yes | true | Cube (48-71) |

---

## Previous Updates

### 2026-01-22: Sliding Window Task Management

- Implemented sliding window scheme (`PTO_TASK_WINDOW_SIZE = 8K`)
- Task indices wrap around using `PTO_TASK_SLOT(task_id)`
- Window overflow handling based on runtime mode:
  - `BENCHMARK_ONLY`: Advance `window_oldest_pending` to allow TensorMap cleanup
  - `DUMP_GRAPH`: Abort orchestration when window fills
  - `EXECUTE/SIMULATE`: Stall until tasks complete

### 2026-01-21: TensorMap Optimization

- Replaced simple hash with MurmurHash-style function
- Added memory pool for TensorMapEntry allocations
- Simplified insert/lookup with in-place stale entry overwriting
- Improved throughput from 3.5M to 3.77M tasks/ms

### 2026-01-20: Binary Expansion Residual Handling

- Fixed issue where small sequence lengths produced 0 tasks
- Added residual loop in `apply_binary_expansion()` for `num_tiles < MIN_NUM_TILES`

### 2026-01-19: Sequence Length Configuration

- Changed benchmark config from `num_tiles` to `seq_len` (more intuitive)
- Conversion: `num_tiles = seq_len // TILE_ROWS` (TILE_ROWS = 32)
- Default range: 1024-16384 tokens, step 1024

### 2026-01-18: Record & Replay Archival

- Archived Record & Replay feature to `pto_record_replay_archived.h`
- Removed from active runtime due to conflicts with sliding window
- Simplified task table to single-level array
