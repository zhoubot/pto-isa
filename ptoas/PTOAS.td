//===- PTOAS.td ------------------------------------------------*- tablegen -*-===//
//
// Draft MLIR dialect for PTO Assembly (PTO-AS).
//
// This file is meant to capture the assembly surface form and the type/attribute
// model so it stays consistent with `include/pto/*` C++ templates.
//
// NOTE: This is a spec-only prototype (TableGen). A full implementation would
// add C++ dialect/type defs and a lowering pipeline.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def PTOAS_Dialect : Dialect {
  let name = "pto";
  let summary = "PTO Tile Lib assembly dialect (prototype)";
  let description = [{
    PTO dialect models PTO Tile Lib intrinsics in a destination-passing style.
  }];
}

// ---- Attributes (draft) ----

def PTO_LayoutAttr : EnumAttr<PTOAS_Dialect, "Layout", "GlobalTensor Layout hint",
  [EnumAttrCase<"ND", 0>,
   EnumAttrCase<"DN", 1>,
   EnumAttrCase<"NZ", 2>,
   EnumAttrCase<"SCALE", 3>,
   EnumAttrCase<"MX_A_ND", 4>,
   EnumAttrCase<"MX_A_DN", 5>,
   EnumAttrCase<"MX_A_ZZ", 6>,
   EnumAttrCase<"MX_B_ND", 7>,
   EnumAttrCase<"MX_B_DN", 8>,
   EnumAttrCase<"MX_B_NN", 9>,
   EnumAttrCase<"NC1HWC0", 10>,
   EnumAttrCase<"NCHW", 11>,
   EnumAttrCase<"NHWC", 12>,
   EnumAttrCase<"FRACTAL_Z", 13>,
   EnumAttrCase<"FRACTAL_Z_S16S8", 14>,
   EnumAttrCase<"MAX", 15>]> {
  let cppNamespace = "::pto";
}

def PTO_TileTypeAttr : EnumAttr<PTOAS_Dialect, "TileType", "Tile storage class (pto::TileType)",
  [EnumAttrCase<"Vec", 0>,
   EnumAttrCase<"Mat", 1>,
   EnumAttrCase<"Left", 2>,
   EnumAttrCase<"Right", 3>,
   EnumAttrCase<"Acc", 4>,
   EnumAttrCase<"Bias", 5>,
   EnumAttrCase<"Scaling", 6>,
   EnumAttrCase<"ScaleLeft", 7>,
   EnumAttrCase<"ScaleRight", 8>]> {
  let cppNamespace = "::pto";
}

def PTO_BLayoutAttr : EnumAttr<PTOAS_Dialect, "BLayout", "Tile base layout (pto::BLayout)",
  [EnumAttrCase<"RowMajor", 0>,
   EnumAttrCase<"ColMajor", 1>]> {
  let cppNamespace = "::pto";
}

def PTO_SLayoutAttr : EnumAttr<PTOAS_Dialect, "SLayout", "Tile boxed/fractal layout (pto::SLayout)",
  [EnumAttrCase<"NoneBox", 0>,
   EnumAttrCase<"RowMajor", 1>,
   EnumAttrCase<"ColMajor", 2>]> {
  let cppNamespace = "::pto";
}

def PTO_PadValueAttr : EnumAttr<PTOAS_Dialect, "PadValue", "Tile padding policy (pto::PadValue)",
  [EnumAttrCase<"Null", 0>,
   EnumAttrCase<"Zero", 1>,
   EnumAttrCase<"Max", 2>,
   EnumAttrCase<"Min", 3>]> {
  let cppNamespace = "::pto";
}

def PTO_CmpModeAttr : EnumAttr<PTOAS_Dialect, "CmpMode", "Compare mode (pto::CmpMode)",
  [EnumAttrCase<"EQ", 0>,
   EnumAttrCase<"NE", 1>,
   EnumAttrCase<"LT", 2>,
   EnumAttrCase<"LE", 3>,
   EnumAttrCase<"GT", 4>,
   EnumAttrCase<"GE", 5>]> {
  let cppNamespace = "::pto";
}

// Mirrors include/pto/common/event.hpp (excluding OP_COUNT).
def PTO_OpAttr : EnumAttr<PTOAS_Dialect, "Op", "PTO op code (pto::Op)",
  [EnumAttrCase<"TLOAD", 0>,
   EnumAttrCase<"TSTORE_VEC", 1>,
   EnumAttrCase<"SCALAR", 2>,
   EnumAttrCase<"TRESHAPE", 3>,
   EnumAttrCase<"VECTOR", 4>,
   EnumAttrCase<"TADD", 5>,
   EnumAttrCase<"TADDS", 6>,
   EnumAttrCase<"TSUB", 7>,
   EnumAttrCase<"TMUL", 8>,
   EnumAttrCase<"TMULS", 9>,
   EnumAttrCase<"TDIV", 10>,
   EnumAttrCase<"TDIVS", 11>,
   EnumAttrCase<"TMIN", 12>,
   EnumAttrCase<"TMINS", 13>,
   EnumAttrCase<"TMAX", 14>,
   EnumAttrCase<"TAND", 15>,
   EnumAttrCase<"TOR", 16>,
   EnumAttrCase<"TSEL", 17>,
   EnumAttrCase<"TEXP", 18>,
   EnumAttrCase<"TSELS", 19>,
   EnumAttrCase<"TSQRT", 20>,
   EnumAttrCase<"TRSQRT", 21>,
   EnumAttrCase<"TEXPANDS", 22>,
   EnumAttrCase<"TPARTADD", 23>,
   EnumAttrCase<"TPARTMAX", 24>,
   EnumAttrCase<"TPARTMIN", 25>,
   EnumAttrCase<"TCMPS", 26>,
   EnumAttrCase<"TMRGSORT", 27>,
   EnumAttrCase<"TSORT32", 28>,
   EnumAttrCase<"TCI", 29>,
   EnumAttrCase<"TGATHER", 30>,
   EnumAttrCase<"TGATHERB", 31>,
   EnumAttrCase<"TCVT", 32>,
   EnumAttrCase<"TROWSUM", 33>,
   EnumAttrCase<"TROWMAX", 34>,
   EnumAttrCase<"TROWMIN", 35>,
   EnumAttrCase<"TROWEXPAND", 36>,
   EnumAttrCase<"TCOLSUM", 37>,
   EnumAttrCase<"TCOLMAX", 38>,
   EnumAttrCase<"TCOLMIN", 39>,
   EnumAttrCase<"TTRANS", 40>,
   EnumAttrCase<"TTRI", 41>,
   EnumAttrCase<"TREM", 42>,
   EnumAttrCase<"TREMS", 43>,
   EnumAttrCase<"TSUBS", 44>,
   EnumAttrCase<"TMAXS", 45>,
   EnumAttrCase<"TLRELU", 46>,
   EnumAttrCase<"TMOV_V2V", 47>,
   EnumAttrCase<"TMOV_V2M", 48>,
   EnumAttrCase<"TEXTRACT_V2M", 49>,
   EnumAttrCase<"TMOV_M2B", 50>,
   EnumAttrCase<"TMOV_M2L", 51>,
   EnumAttrCase<"TMOV_M2R", 52>,
   EnumAttrCase<"TMOV_M2S", 53>,
   EnumAttrCase<"TMOV_A2V", 54>,
   EnumAttrCase<"TMOV_A2M", 55>,
   EnumAttrCase<"TSTORE_ACC", 56>,
   EnumAttrCase<"TSTORE_MAT", 57>,
   EnumAttrCase<"TMATMUL", 58>,
   EnumAttrCase<"TMATMUL_MX", 59>,
   EnumAttrCase<"TEXTRACT_M2LR", 60>,
   EnumAttrCase<"TANDS", 61>,
   EnumAttrCase<"TORS", 62>,
   EnumAttrCase<"TSHLS", 63>,
   EnumAttrCase<"TSHRS", 64>,
   EnumAttrCase<"TXOR", 65>,
   EnumAttrCase<"TXORS", 66>,
   EnumAttrCase<"TEXTRACT_A2M", 67>,
   EnumAttrCase<"TINSERT_A2M", 68>]> {
  let cppNamespace = "::pto";
}

// ---- Types (draft, textual-only) ----
// Full MLIR TypeDef integration is intentionally omitted here; we specify the
// textual spelling contract that the PTO-AS docs reference.
//
// !pto.tile<loc=Vec, dtype=f16, rows=16, cols=16, blayout=RowMajor, valid=16x16, slayout=NoneBox, fractal=512, pad=Null>
// !pto.tensor<dtype=f16, shape=[1,1,1,16,16], stride=[1,1,1,16,1], layout=ND>
// !pto.event<src=#pto.op<TLOAD>, dst=#pto.op<TADD>>

// ---- Ops (draft) ----

class PTO_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTOAS_Dialect, mnemonic, traits>;

// PTO ops are side-effecting (write to destination tiles / global memory).
def PTO_TAddOp : PTO_Op<"tadd", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Elementwise add (DPS)";
  let arguments = (ins AnyType:$dst, AnyType:$src0, AnyType:$src1);
  let assemblyFormat = "$dst `,` $src0 `,` $src1 attr-dict `:` `(` type($dst) `,` type($src0) `,` type($src1) `)`";
}

def PTO_TLoadOp : PTO_Op<"tload", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Load GlobalTensor -> Tile (DPS)";
  let arguments = (ins AnyType:$dst, AnyType:$src, Index:$r0, Index:$c0);
  let assemblyFormat = "$dst `,` $src `[` $r0 `,` $c0 `]` attr-dict `:` `(` type($dst) `,` type($src) `,` type($r0) `,` type($c0) `)`";
}

def PTO_TStoreOp : PTO_Op<"tstore", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Store Tile -> GlobalTensor (DPS)";
  let arguments = (ins AnyType:$dst, Index:$r0, Index:$c0, AnyType:$src);
  let assemblyFormat = "$dst `[` $r0 `,` $c0 `]` `,` $src attr-dict `:` `(` type($dst) `,` type($r0) `,` type($c0) `,` type($src) `)`";
}

def PTO_MGatherOp : PTO_Op<"mgather", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Gather GlobalTensor -> Tile (DPS)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$idx);
  let assemblyFormat = "$dst `,` $src `,` $idx attr-dict `:` `(` type($dst) `,` type($src) `,` type($idx) `)`";
}

def PTO_MScatterOp : PTO_Op<"mscatter", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Scatter Tile -> GlobalTensor (DPS)";
  let arguments = (ins AnyType:$dst, AnyType:$src, AnyType:$idx);
  let assemblyFormat = "$dst `,` $src `,` $idx attr-dict `:` `(` type($dst) `,` type($src) `,` type($idx) `)`";
}

def PTO_TMrgsortOp : PTO_Op<"tmrgsort", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "Merge sort (DPS, minimal 2-list form)";
  let arguments = (ins AnyType:$dst, AnyType:$executed, AnyType:$tmp, AnyType:$src0, AnyType:$src1);
  let assemblyFormat = "$dst `,` $executed `,` $tmp `,` $src0 `,` $src1 attr-dict `:` `(` type($dst) `,` type($executed) `,` type($tmp) `,` type($src0) `,` type($src1) `)`";
}

// Auto-generated stubs for the remaining intrinsics in include/pto/common/pto_instr.hpp.
include "PTOASOps.td"
