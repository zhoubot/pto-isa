<p align="center">
  <img src="../figures/pto_logo.svg" alt="PTO Tile Lib" width="180" />
</p>

# PTO ISA Reference

This directory contains the per-instruction reference for the PTO Tile Lib ISA.

- Source of truth (C++ intrinsics): `include/pto/common/pto_instr.hpp`
- Common conventions (operands, events, modifiers): `docs/isa/conventions.md`


## Elementwise (Tile-Tile)
- `TADD`: `docs/isa/TADD.md`
- `TABS`: `docs/isa/TABS.md`
- `TSUB`: `docs/isa/TSUB.md`
- `TMUL`: `docs/isa/TMUL.md`
- `TDIV`: `docs/isa/TDIV.md`
- `TREM`: `docs/isa/TREM.md`
- `TSHL`: `docs/isa/TSHL.md`
- `TSHR`: `docs/isa/TSHR.md`
- `TAND`: `docs/isa/TAND.md`
- `TOR`: `docs/isa/TOR.md`
- `TXOR`: `docs/isa/TXOR.md`
- `TMIN`: `docs/isa/TMIN.md`
- `TMAX`: `docs/isa/TMAX.md`
- `TEXP`: `docs/isa/TEXP.md`
- `TLOG`: `docs/isa/TLOG.md`
- `TSQRT`: `docs/isa/TSQRT.md`
- `TRSQRT`: `docs/isa/TRSQRT.md`
- `TRECIP`: `docs/isa/TRECIP.md`
- `TNEG`: `docs/isa/TNEG.md`
- `TNOT`: `docs/isa/TNOT.md`
- `TRELU`: `docs/isa/TRELU.md`
- `TPRELU`: `docs/isa/TPRELU.md`
- `TADDC`: `docs/isa/TADDC.md`
- `TSUBC`: `docs/isa/TSUBC.md`
- `TSEL`: `docs/isa/TSEL.md`
- `TCMP`: `docs/isa/TCMP.md`
- `TCVT`: `docs/isa/TCVT.md`

## Tile-Scalar / Tile-Immediate
- `TADDS`: `docs/isa/TADDS.md`
- `TSUBS`: `docs/isa/TSUBS.md`
- `TDIVS`: `docs/isa/TDIVS.md`
- `TMULS`: `docs/isa/TMULS.md`
- `TREMS`: `docs/isa/TREMS.md`
- `TMAXS`: `docs/isa/TMAXS.md`
- `TMINS`: `docs/isa/TMINS.md`
- `TANDS`: `docs/isa/TANDS.md`
- `TORS`: `docs/isa/TORS.md`
- `TXORS`: `docs/isa/TXORS.md`
- `TCMPS`: `docs/isa/TCMPS.md`
- `TEXPANDS`: `docs/isa/TEXPANDS.md`
- `TSELS`: `docs/isa/TSELS.md`
- `TLRELU`: `docs/isa/TLRELU.md`
- `TADDSC`: `docs/isa/TADDSC.md`
- `TSUBSC`: `docs/isa/TSUBSC.md`

## Axis Reduce / Expand
- `TROWSUM`: `docs/isa/TROWSUM.md`
- `TROWMAX`: `docs/isa/TROWMAX.md`
- `TROWMIN`: `docs/isa/TROWMIN.md`
- `TROWEXPAND`: `docs/isa/TROWEXPAND.md`
- `TROWEXPANDDIV`: `docs/isa/TROWEXPANDDIV.md`
- `TROWEXPANDMUL`: `docs/isa/TROWEXPANDMUL.md`
- `TROWEXPANDSUB`: `docs/isa/TROWEXPANDSUB.md`
- `TROWEXPANDADD`: `docs/isa/TROWEXPANDADD.md`
- `TROWEXPANDMAX`: `docs/isa/TROWEXPANDMAX.md`
- `TROWEXPANDMIN`: `docs/isa/TROWEXPANDMIN.md`
- `TCOLSUM`: `docs/isa/TCOLSUM.md`
- `TCOLMAX`: `docs/isa/TCOLMAX.md`
- `TCOLMIN`: `docs/isa/TCOLMIN.md`
- `TCOLEXPAND`: `docs/isa/TCOLEXPAND.md`
- `TCOLEXPANDDIV`: `docs/isa/TCOLEXPANDDIV.md`
- `TCOLEXPANDMUL`: `docs/isa/TCOLEXPANDMUL.md`
- `TCOLEXPANDSUB`: `docs/isa/TCOLEXPANDSUB.md`

## Memory (GM <-> Tile)
- `TLOAD`: `docs/isa/TLOAD.md`
- `TSTORE`: `docs/isa/TSTORE.md`
- `TSTORE_FP`: `docs/isa/TSTORE_FP.md`
- `MGATHER`: `docs/isa/MGATHER.md`
- `MSCATTER`: `docs/isa/MSCATTER.md`

## Matrix Multiply
- `TMATMUL`: `docs/isa/TMATMUL.md`
- `TMATMUL_MX`: `docs/isa/TMATMUL_MX.md`
- `TMATMUL_ACC`: `docs/isa/TMATMUL_ACC.md`
- `TMATMUL_BIAS`: `docs/isa/TMATMUL_BIAS.md`

## Data Movement / Layout
- `TMOV`: `docs/isa/TMOV.md`
- `TMOV_FP`: `docs/isa/TMOV_FP.md`
- `TTRANS`: `docs/isa/TTRANS.md`
- `TEXTRACT`: `docs/isa/TEXTRACT.md`
- `TRESHAPE`: `docs/isa/TRESHAPE.md`
- `TASSIGN`: `docs/isa/TASSIGN.md`

## Padding
- `TFILLPAD`: `docs/isa/TFILLPAD.md`
- `TFILLPAD_INPLACE`: `docs/isa/TFILLPAD_INPLACE.md`
- `TFILLPAD_EXPAND`: `docs/isa/TFILLPAD_EXPAND.md`

## Complex
- `TTRI`: `docs/isa/TTRI.md`
- `TCI`: `docs/isa/TCI.md`
- `TGATHER`: `docs/isa/TGATHER.md`
- `TGATHERB`: `docs/isa/TGATHERB.md`
- `TSCATTER`: `docs/isa/TSCATTER.md`
- `TSORT32`: `docs/isa/TSORT32.md`
- `TMRGSORT`: `docs/isa/TMRGSORT.md`
- `TPARTADD`: `docs/isa/TPARTADD.md`
- `TPARTMAX`: `docs/isa/TPARTMAX.md`
- `TPARTMIN`: `docs/isa/TPARTMIN.md`

## Synchronization
- `TSYNC`: `docs/isa/TSYNC.md`
