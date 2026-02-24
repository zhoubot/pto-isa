# matmul_static_singlecore (CPU simulator)

Static 32x256 @ 256x32 matmul with split-K (BASEK=32, iters=8) + bias.

## Run

```bash
bash examples/ptodsl_ptoas_cpu/matmul_static_singlecore/run.sh
```

Expected:

```text
PASS: CPU-sim RunTMATMULSplitK (static, singlecore)
```
