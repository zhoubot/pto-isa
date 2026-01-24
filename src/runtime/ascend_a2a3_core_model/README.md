# Ascend A2/A3 Core Simulator

Cycle-accurate simulation model of Ascend A2/A3 NPU cores for InCore function execution.

## Architecture

### Cube Core

The Cube Core is optimized for matrix multiplication operations:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  CUBE CORE                                                              │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐             │
│  │  Scalar  │   │ MTE_GM2L1│   │ MTE_L12GM│   │ MTE_L0C  │             │
│  │   Unit   │   │   Pipe   │   │   Pipe   │   │   Pipe   │             │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘             │
│       │              │              │              │                    │
│       │              └──────────────┴──────────────┘                    │
│       │                           │                                     │
│       │              ┌────────────┴────────────┐                        │
│       │              │       CUBE Unit         │                        │
│       │              │  (Matrix Multiply)      │                        │
│       │              └─────────────────────────┘                        │
│       │                                                                 │
│  Memory: GM <-> L1 <-> L0A/L0B/L0C                                     │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pipes:**
- `SCALAR`: Scalar arithmetic, control flow, synchronization
- `MTE_GM2L1`: Global Memory → L1 data transfer
- `MTE_L12GM`: L1 → Global Memory data transfer
- `MTE_L0C`: L0C (Cube output buffer) data movement
- `CUBE`: Matrix multiply unit

### Vector Core

The Vector Core handles element-wise operations and reductions:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  VECTOR CORE                                                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                             │
│  │  Scalar  │   │ MTE_GM2UB│   │ MTE_UB2GM│                             │
│  │   Unit   │   │   Pipe   │   │   Pipe   │                             │
│  └──────────┘   └──────────┘   └──────────┘                             │
│       │              │              │                                   │
│       │              └──────────────┘                                   │
│       │                     │                                           │
│       │         ┌───────────┴───────────┐                               │
│       │         │      Vector Unit      │                               │
│       │         │  (Element-wise ops,   │                               │
│       │         │   Reductions, etc.)   │                               │
│       │         └───────────────────────┘                               │
│       │                                                                 │
│  Memory: GM <-> UB (Unified Buffer)                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Pipes:**
- `SCALAR`: Scalar arithmetic, control flow, synchronization
- `MTE_GM2UB`: Global Memory → Unified Buffer transfer
- `MTE_UB2GM`: Unified Buffer → Global Memory transfer
- `VECTOR`: Vector compute unit

## Synchronization

The simulator supports standard Ascend synchronization primitives:

| Instruction | Description |
|-------------|-------------|
| `SET_FLAG(n)` | Signal flag `n` is set (source pipe done) |
| `WAIT_FLAG(n)` | Wait for flag `n` to be set |
| `pipe_barrier()` | Synchronize all pipes to maximum cycle |

## Cycle Costs

Default latencies (configurable in `a2a3_core_model.h`):

| Operation | Latency (cycles) |
|-----------|------------------|
| MTE GM↔L1 | 100 + size/256 |
| MTE GM↔UB | 80 + size/256 |
| MTE L0C | 20 |
| CUBE MatMul | 50 |
| Vector Binary | 10 |
| Vector Unary | 10 |
| Vector Reduce | 20 |
| Vector Activation | 15 |
| Scalar | 1 |

## Usage

### Building

```bash
cd src/runtime/ascend_a2a3_core
make
```

### Running Tests

```bash
make test
```

### API Example

```c
#include "a2a3_core_model.h"
#include "a2a3_incore_sim.h"

// Create simulator
IncoreSimulator* sim = a2a3_incore_sim_create();

// Register a function
const char* instructions[] = {
    "DataCopy(x, input, 4096);",
    "Mul(y, x, x, 4096);",
    "ReduceSum(sum, y, 4096);",
    "pipe_barrier();",
    NULL
};

int func_id = a2a3_incore_sim_register(sim, "my_function",
                                        CORE_TYPE_VECTOR,
                                        instructions, 4,
                                        32, 128);

// Simulate and get cycle count
int64_t cycles = a2a3_incore_sim_execute(sim, func_id);
printf("Function took %lld cycles\n", cycles);

// Or use heuristic for quick estimates
int64_t estimated = a2a3_get_incore_cycle_cost("rmsnorm_tile", 32*128);

// Cleanup
a2a3_incore_sim_destroy(sim);
```

## Integration with PTO Runtime

The core simulator integrates with the PTO runtime for `ascend_a2a3_sim` platform:

1. **Code Generation**: Same as `ascend_a2a3` target (generates Ascend instructions)
2. **Simulation**: Uses core model to estimate InCore function cycle counts
3. **Task Scheduling**: Cycle costs drive task scheduling in orchestration

## Files

| File | Description |
|------|-------------|
| `a2a3_core_model.h` | Core model data structures and API |
| `a2a3_core_model.c` | Core model implementation |
| `a2a3_incore_sim.h` | InCore function simulator API |
| `a2a3_incore_sim.c` | InCore simulator implementation |
| `test_core_sim.c` | Test program |
| `Makefile` | Build system |

## Future Enhancements

- [ ] More accurate MTE latency modeling (burst transfers, bank conflicts)
- [ ] L1/UB cache simulation
- [ ] Power estimation
- [ ] Integration with actual Ascend instruction encoding
