//===- PTOASOps.td --------------------------------------------*- tablegen -*-===//
//
// Auto-generated spec-only op stubs for PTO-AS.
//
// Regenerate with:
//   python3 ptoas/tools/gen_ptoas_ops_td.py
//
// Source of truth for instruction names:
//   include/pto/common/pto_instr.hpp
//
// NOTE: These are intentionally generic (variadic operands).
//       A full dialect would model operands/results/types precisely.
//
//===----------------------------------------------------------------------===//

def PTO_TabsOp : PTO_Op<"tabs", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TABS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TaddcOp : PTO_Op<"taddc", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TADDC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TaddsOp : PTO_Op<"tadds", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TADDS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TaddscOp : PTO_Op<"taddsc", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TADDSC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TandOp : PTO_Op<"tand", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TAND (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TandsOp : PTO_Op<"tands", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TANDS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TassignOp : PTO_Op<"tassign", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TASSIGN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TciOp : PTO_Op<"tci", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCI (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcmpOp : PTO_Op<"tcmp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCMP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcmpsOp : PTO_Op<"tcmps", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCMPS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolexpandOp : PTO_Op<"tcolexpand", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLEXPAND (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolexpanddivOp : PTO_Op<"tcolexpanddiv", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLEXPANDDIV (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolexpandexpdifOp : PTO_Op<"tcolexpandexpdif", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLEXPANDEXPDIF (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolexpandmulOp : PTO_Op<"tcolexpandmul", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLEXPANDMUL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolexpandsubOp : PTO_Op<"tcolexpandsub", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLEXPANDSUB (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolmaxOp : PTO_Op<"tcolmax", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLMAX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolminOp : PTO_Op<"tcolmin", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLMIN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcolsumOp : PTO_Op<"tcolsum", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCOLSUM (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TcvtOp : PTO_Op<"tcvt", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TCVT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TdivOp : PTO_Op<"tdiv", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TDIV (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TdivsOp : PTO_Op<"tdivs", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TDIVS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TexpOp : PTO_Op<"texp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TEXP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TexpandsOp : PTO_Op<"texpands", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TEXPANDS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TextractOp : PTO_Op<"textract", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TEXTRACT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TextractFpOp : PTO_Op<"textract_fp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TEXTRACT_FP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TfillpadOp : PTO_Op<"tfillpad", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TFILLPAD (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TfillpadExpandOp : PTO_Op<"tfillpad_expand", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TFILLPAD_EXPAND (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TfillpadInplaceOp : PTO_Op<"tfillpad_inplace", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TFILLPAD_INPLACE (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TgatherOp : PTO_Op<"tgather", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TGATHER (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TgatherbOp : PTO_Op<"tgatherb", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TGATHERB (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TinsertOp : PTO_Op<"tinsert", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TINSERT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TinsertFpOp : PTO_Op<"tinsert_fp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TINSERT_FP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TlogOp : PTO_Op<"tlog", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TLOG (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TlreluOp : PTO_Op<"tlrelu", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TLRELU (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmatmulOp : PTO_Op<"tmatmul", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMATMUL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmatmulAccOp : PTO_Op<"tmatmul_acc", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMATMUL_ACC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmatmulBiasOp : PTO_Op<"tmatmul_bias", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMATMUL_BIAS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmatmulMxOp : PTO_Op<"tmatmul_mx", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMATMUL_MX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmaxOp : PTO_Op<"tmax", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMAX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmaxsOp : PTO_Op<"tmaxs", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMAXS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TminOp : PTO_Op<"tmin", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMIN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TminsOp : PTO_Op<"tmins", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMINS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmovOp : PTO_Op<"tmov", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMOV (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmovFpOp : PTO_Op<"tmov_fp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMOV_FP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmulOp : PTO_Op<"tmul", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMUL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TmulsOp : PTO_Op<"tmuls", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TMULS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TnegOp : PTO_Op<"tneg", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TNEG (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TnotOp : PTO_Op<"tnot", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TNOT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TorOp : PTO_Op<"tor", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TOR (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TorsOp : PTO_Op<"tors", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TORS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TpartaddOp : PTO_Op<"tpartadd", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPARTADD (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TpartmaxOp : PTO_Op<"tpartmax", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPARTMAX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TpartminOp : PTO_Op<"tpartmin", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPARTMIN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TprefetchOp : PTO_Op<"tprefetch", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPREFETCH (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TpreluOp : PTO_Op<"tprelu", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPRELU (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TprintOp : PTO_Op<"tprint", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TPRINT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrecipOp : PTO_Op<"trecip", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TRECIP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TreluOp : PTO_Op<"trelu", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TRELU (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TremOp : PTO_Op<"trem", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TREM (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TremsOp : PTO_Op<"trems", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TREMS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TreshapeOp : PTO_Op<"treshape", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TRESHAPE (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandOp : PTO_Op<"trowexpand", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPAND (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandaddOp : PTO_Op<"trowexpandadd", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDADD (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpanddivOp : PTO_Op<"trowexpanddiv", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDDIV (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandexpdifOp : PTO_Op<"trowexpandexpdif", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDEXPDIF (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandmaxOp : PTO_Op<"trowexpandmax", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDMAX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandminOp : PTO_Op<"trowexpandmin", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDMIN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandmulOp : PTO_Op<"trowexpandmul", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDMUL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowexpandsubOp : PTO_Op<"trowexpandsub", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWEXPANDSUB (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowmaxOp : PTO_Op<"trowmax", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWMAX (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowminOp : PTO_Op<"trowmin", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWMIN (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrowsumOp : PTO_Op<"trowsum", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TROWSUM (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TrsqrtOp : PTO_Op<"trsqrt", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TRSQRT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TscatterOp : PTO_Op<"tscatter", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSCATTER (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TselOp : PTO_Op<"tsel", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSEL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TselsOp : PTO_Op<"tsels", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSELS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TshlOp : PTO_Op<"tshl", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSHL (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TshlsOp : PTO_Op<"tshls", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSHLS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TshrOp : PTO_Op<"tshr", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSHR (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TshrsOp : PTO_Op<"tshrs", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSHRS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_Tsort32Op : PTO_Op<"tsort32", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSORT32 (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsqrtOp : PTO_Op<"tsqrt", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSQRT (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TstoreFpOp : PTO_Op<"tstore_fp", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSTORE_FP (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsubOp : PTO_Op<"tsub", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSUB (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsubcOp : PTO_Op<"tsubc", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSUBC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsubsOp : PTO_Op<"tsubs", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSUBS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsubscOp : PTO_Op<"tsubsc", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSUBSC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TsyncOp : PTO_Op<"tsync", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TSYNC (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TtransOp : PTO_Op<"ttrans", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TTRANS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TtriOp : PTO_Op<"ttri", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TTRI (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TxorOp : PTO_Op<"txor", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TXOR (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}

def PTO_TxorsOp : PTO_Op<"txors", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "TXORS (auto-generated stub)";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "$operands attr-dict";
}
