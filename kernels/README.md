# Kernels

This directory contains runnable kernel/operator examples.

## Where to start

- Python kernels (recommended): `kernels/python/`
  - Flow: Python -> PTO-AS -> `ptoas` -> build/run (CPU ref; optional Ascend NPU)
  - Entry point: `kernels/python/run_regression.py`
- Custom examples: `kernels/custom/`

## Notes

- This public repo intentionally does **not** include the large `kernels/manual/` tree from the private repo.
- For NPU runs, set `ASCEND_HOME_PATH` (e.g. `$HOME/Ascend/ascend-toolkit/latest`) and use `bin/ptoas`.
