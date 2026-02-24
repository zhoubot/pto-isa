# Generate a smaller matmul-dynbatch module for CPU simulator runs.
#
# Upstream PTODSL example defaults are larger; we shrink for speed.

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
builder_path = ROOT / "PTODSL" / "examples" / "aot" / "matmul_dynbatch_multicore" / "matmul_builder.py"

spec = importlib.util.spec_from_file_location("ptodsl_matmul_dynbatch_builder", builder_path)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

m = mod.build(M=32, K=64, N=32, validM=32, validK=64, validN=32, BASEK=32)
print(m)
