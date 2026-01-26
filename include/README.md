# include/

Public C/C++ headers for PTO Tile Lib (primarily header-only, template-based). Upper-layer frameworks or operator code can include these headers to emit PTO ISA Tile-level operations.

## Quick Start

Include the unified entry header:

```cpp
#include <pto/pto-inst.hpp>
```

`pto/pto-inst.hpp` selects the appropriate backend (CPU simulation/stub or NPU implementation) based on build configuration. See `include/pto/README.md` for details.

## Layout

- `include/pto/`: Public PTO ISA API and backend implementations (common / cpu / npu)

## Related Docs

- ISA guide: `docs/README.md`
- Getting started: `docs/getting-started.md`

## PTO Instruction Implementation Status (CPU / A2 / A3 / A5)

This table tracks per-instruction backend availability:

- **CPU**: `__CPU_SIM` (CPU simulation backend).
- **A2 (Ascend 910B) / A3 (Ascend 910C)**: share the `include/pto/npu/a2a3/` implementation today (so the status is identical for both columns).
- **A5 (Ascend 950)**: uses the `include/pto/npu/a5/` implementation.
- **TODO** means the instruction is part of the public API but the backend implementation is not available yet.

| Instruction | CPU | A2 | A3 | A5 |
|---|---:|---:|---:|---:|
| [`MGATHER`](../docs/isa/MGATHER.md) | Yes | TODO | TODO | TODO |
| [`MSCATTER`](../docs/isa/MSCATTER.md) | Yes | TODO | TODO | TODO |
| [`TABS`](../docs/isa/TABS.md) | TODO | TODO | TODO | TODO |
| [`TADD`](../docs/isa/TADD.md) | Yes | Yes | Yes | Yes |
| [`TADDC`](../docs/isa/TADDC.md) | Yes | TODO | TODO | TODO |
| [`TADDS`](../docs/isa/TADDS.md) | Yes | Yes | Yes | Yes |
| [`TADDSC`](../docs/isa/TADDSC.md) | Yes | TODO | TODO | TODO |
| [`TAND`](../docs/isa/TAND.md) | Yes | TODO | TODO | TODO |
| [`TANDS`](../docs/isa/TANDS.md) | Yes | TODO | TODO | TODO |
| [`TASSIGN`](../docs/isa/TASSIGN.md) | Yes | Yes | Yes | Yes |
| [`TCI`](../docs/isa/TCI.md) | Yes | Yes | Yes | Yes |
| [`TCMP`](../docs/isa/TCMP.md) | TODO | TODO | TODO | TODO |
| [`TCMPS`](../docs/isa/TCMPS.md) | Yes | Yes | Yes | Yes |
| [`TCOLEXPAND`](../docs/isa/TCOLEXPAND.md) | Yes | TODO | TODO | TODO |
| [`TCOLMAX`](../docs/isa/TCOLMAX.md) | Yes | Yes | Yes | Yes |
| [`TCOLMIN`](../docs/isa/TCOLMIN.md) | TODO | TODO | TODO | TODO |
| [`TCOLSUM`](../docs/isa/TCOLSUM.md) | Yes | Yes | Yes | Yes |
| [`TCVT`](../docs/isa/TCVT.md) | Yes | Yes | Yes | Yes |
| [`TDIV`](../docs/isa/TDIV.md) | Yes | Yes | Yes | Yes |
| [`TDIVS`](../docs/isa/TDIVS.md) | Yes | Yes | Yes | Yes |
| [`TEXP`](../docs/isa/TEXP.md) | Yes | Yes | Yes | Yes |
| [`TEXPANDS`](../docs/isa/TEXPANDS.md) | Yes | Yes | Yes | Yes |
| [`TEXTRACT`](../docs/isa/TEXTRACT.md) | Yes | Yes | Yes | Yes |
| [`TGATHER`](../docs/isa/TGATHER.md) | Yes | Yes | Yes | Yes |
| [`TGATHERB`](../docs/isa/TGATHERB.md) | Yes | Yes | Yes | Yes |
| [`TLOAD`](../docs/isa/TLOAD.md) | Yes | Yes | Yes | Yes |
| [`TLOG`](../docs/isa/TLOG.md) | TODO | TODO | TODO | TODO |
| [`TLRELU`](../docs/isa/TLRELU.md) | Yes | TODO | TODO | TODO |
| [`TMATMUL`](../docs/isa/TMATMUL.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL_ACC`](../docs/isa/TMATMUL_ACC.md) | Yes | Yes | Yes | Yes |
| [`TMATMUL_BIAS`](../docs/isa/TMATMUL_BIAS.md) | Yes | Yes | Yes | Yes |
| [`TMAX`](../docs/isa/TMAX.md) | Yes | Yes | Yes | Yes |
| [`TMAXS`](../docs/isa/TMAXS.md) | Yes | TODO | TODO | TODO |
| [`TMIN`](../docs/isa/TMIN.md) | TODO | TODO | TODO | TODO |
| [`TMINS`](../docs/isa/TMINS.md) | Yes | Yes | Yes | Yes |
| [`TMOV`](../docs/isa/TMOV.md) | Yes | Yes | Yes | Yes |
| [`TMOV_FP`](../docs/isa/TMOV_FP.md) | TODO | TODO | TODO | TODO |
| [`TMRGSORT`](../docs/isa/TMRGSORT.md) | Yes | Yes | Yes | Yes |
| [`TMUL`](../docs/isa/TMUL.md) | Yes | Yes | Yes | Yes |
| [`TMULS`](../docs/isa/TMULS.md) | Yes | Yes | Yes | Yes |
| [`TNEG`](../docs/isa/TNEG.md) | Yes | TODO | TODO | TODO |
| [`TNOT`](../docs/isa/TNOT.md) | Yes | TODO | TODO | TODO |
| [`TOR`](../docs/isa/TOR.md) | Yes | TODO | TODO | TODO |
| [`TORS`](../docs/isa/TORS.md) | Yes | TODO | TODO | TODO |
| [`TPARTADD`](../docs/isa/TPARTADD.md) | Yes | Yes | Yes | Yes |
| [`TPARTMAX`](../docs/isa/TPARTMAX.md) | Yes | Yes | Yes | Yes |
| [`TPARTMIN`](../docs/isa/TPARTMIN.md) | Yes | Yes | Yes | Yes |
| [`TPRELU`](../docs/isa/TPRELU.md) | Yes | TODO | TODO | TODO |
| [`TRECIP`](../docs/isa/TRECIP.md) | TODO | TODO | TODO | TODO |
| [`TRELU`](../docs/isa/TRELU.md) | Yes | TODO | TODO | TODO |
| [`TREM`](../docs/isa/TREM.md) | Yes | TODO | TODO | TODO |
| [`TREMS`](../docs/isa/TREMS.md) | Yes | TODO | TODO | TODO |
| [`TRESHAPE`](../docs/isa/TRESHAPE.md) | TODO | TODO | TODO | TODO |
| [`TROWEXPAND`](../docs/isa/TROWEXPAND.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDDIV`](../docs/isa/TROWEXPANDDIV.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDMUL`](../docs/isa/TROWEXPANDMUL.md) | Yes | Yes | Yes | Yes |
| [`TROWEXPANDSUB`](../docs/isa/TROWEXPANDSUB.md) | Yes | Yes | Yes | Yes |
| [`TROWMAX`](../docs/isa/TROWMAX.md) | Yes | Yes | Yes | Yes |
| [`TROWMIN`](../docs/isa/TROWMIN.md) | TODO | TODO | TODO | TODO |
| [`TROWSUM`](../docs/isa/TROWSUM.md) | Yes | Yes | Yes | Yes |
| [`TRSQRT`](../docs/isa/TRSQRT.md) | Yes | Yes | Yes | Yes |
| [`TSCATTER`](../docs/isa/TSCATTER.md) | TODO | TODO | TODO | TODO |
| [`TSEL`](../docs/isa/TSEL.md) | Yes | Yes | Yes | Yes |
| [`TSELS`](../docs/isa/TSELS.md) | TODO | TODO | TODO | TODO |
| [`TSHL`](../docs/isa/TSHL.md) | Yes | TODO | TODO | TODO |
| [`TSHR`](../docs/isa/TSHR.md) | Yes | TODO | TODO | TODO |
| [`TSORT32`](../docs/isa/TSORT32.md) | Yes | Yes | Yes | Yes |
| [`TSQRT`](../docs/isa/TSQRT.md) | Yes | Yes | Yes | Yes |
| [`TSTORE`](../docs/isa/TSTORE.md) | Yes | Yes | Yes | Yes |
| [`TSTORE_FP`](../docs/isa/TSTORE_FP.md) | TODO | TODO | TODO | TODO |
| [`TSUB`](../docs/isa/TSUB.md) | Yes | Yes | Yes | Yes |
| [`TSUBC`](../docs/isa/TSUBC.md) | Yes | TODO | TODO | TODO |
| [`TSUBS`](../docs/isa/TSUBS.md) | Yes | TODO | TODO | TODO |
| [`TSUBSC`](../docs/isa/TSUBSC.md) | Yes | TODO | TODO | TODO |
| [`TSYNC`](../docs/isa/TSYNC.md) | TODO | TODO | TODO | TODO |
| [`TTRANS`](../docs/isa/TTRANS.md) | Yes | Yes | Yes | Yes |
| [`TXOR`](../docs/isa/TXOR.md) | Yes | TODO | TODO | TODO |
| [`TXORS`](../docs/isa/TXORS.md) | Yes | TODO | TODO | TODO |
