# relu_dynamic_multicore (CPU simulator)

PTODSL example lowered by PTOAS and executed on CPU via PTO-ISA.

## Run

```bash
PTO_CPU_MAX_THREADS=16 bash examples/ptodsl_ptoas_cpu/relu_dynamic_multicore/run.sh
```

Expected:

```text
PASS: CPU-sim sync_kernel_dyn (relu, multicore)
```
