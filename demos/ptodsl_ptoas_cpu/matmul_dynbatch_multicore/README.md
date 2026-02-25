# matmul_dynbatch_multicore (CPU simulator)

End-to-end dynamic batch matmul on CPU simulator.

This example uses a **reduced problem size** for fast CPU runs:

- M=32, K=64, N=32, BASEK=32
- runtime batch=8

## Run

```bash
PTO_CPU_MAX_THREADS=16 bash examples/ptodsl_ptoas_cpu/matmul_dynbatch_multicore/run.sh
```

Expected:

```text
PASS: CPU-sim RunTMATMULSplitK (dynbatch, multicore)
```
