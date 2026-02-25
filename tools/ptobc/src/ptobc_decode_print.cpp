#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"
#include "ptobc/leb128.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/Value.h>
#include <mlir/Parser/Parser.h>

#include <llvm/Support/raw_ostream.h>

#include <stdexcept>
#include <unordered_map>

namespace ptobc {

struct Reader {
  const uint8_t* p;
  const uint8_t* end;

  uint8_t readU8() {
    if (p >= end) throw std::runtime_error("EOF");
    return *p++;
  }
  uint16_t readU16LE() {
    uint16_t lo = readU8();
    uint16_t hi = readU8();
    return lo | (hi << 8);
  }
  uint32_t readU32LE() {
    uint32_t b0 = readU8();
    uint32_t b1 = readU8();
    uint32_t b2 = readU8();
    uint32_t b3 = readU8();
    return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
  }
  uint64_t readULEB() {
    uint64_t v;
    size_t n = readULEB128(p, size_t(end - p), v);
    p += n;
    return v;
  }

  std::vector<uint8_t> readBytes(size_t n) {
    if (size_t(end - p) < n) throw std::runtime_error("EOF");
    std::vector<uint8_t> out(p, p + n);
    p += n;
    return out;
  }
};

static void parseStringsSection(const std::vector<uint8_t>& data, std::vector<std::string>& strings) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  strings.clear();
  strings.reserve(cnt);
  for (uint64_t i = 0; i < cnt; ++i) {
    uint64_t len = r.readULEB();
    auto bs = r.readBytes(len);
    strings.emplace_back(reinterpret_cast<const char*>(bs.data()), bs.size());
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in STRINGS");
}

struct TypeEntry { uint8_t tag; std::string asmStr; };
struct AttrEntry { uint8_t tag; std::string asmStr; };

static void parseTypesSection(const std::vector<uint8_t>& data,
                             const std::vector<std::string>& strings,
                             std::vector<TypeEntry>& types) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  types.clear();
  types.reserve(cnt + 1);
  types.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) throw std::runtime_error("type missing asm");
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) throw std::runtime_error("bad asm_sid");
    types.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in TYPES");
}

static void parseAttrsSection(const std::vector<uint8_t>& data,
                             const std::vector<std::string>& strings,
                             std::vector<AttrEntry>& attrs) {
  Reader r{data.data(), data.data() + data.size()};
  uint64_t cnt = r.readULEB();
  attrs.clear();
  attrs.reserve(cnt + 1);
  attrs.push_back({0, ""});
  for (uint64_t i = 0; i < cnt; ++i) {
    uint8_t tag = r.readU8();
    uint8_t flags = r.readU8();
    if ((flags & 0x1) == 0) throw std::runtime_error("attr missing asm");
    uint64_t sid = r.readULEB();
    if (sid >= strings.size()) throw std::runtime_error("bad asm_sid");
    attrs.push_back({tag, strings[sid]});
  }
  if (r.p != r.end) throw std::runtime_error("trailing bytes in ATTRS");
}

struct BuildCtx {
  mlir::MLIRContext* ctx;
  const std::vector<std::string>* strings;
  const std::vector<TypeEntry>* types;
  const std::vector<AttrEntry>* attrs;

  std::vector<mlir::Value> values;
};

static mlir::Type getType(BuildCtx& bc, uint64_t tid) {
  if (tid >= bc.types->size()) throw std::runtime_error("bad type_id");
  return parseType(*bc.ctx, (*bc.types)[tid].asmStr);
}

static mlir::DictionaryAttr getAttrDict(BuildCtx& bc, uint64_t aid) {
  if (aid == 0) return mlir::DictionaryAttr::get(bc.ctx);
  if (aid >= bc.attrs->size()) throw std::runtime_error("bad attr_id");
  return parseAttrDict(*bc.ctx, (*bc.attrs)[aid].asmStr);
}

static mlir::Region buildRegion(BuildCtx& bc, Reader& r, mlir::Operation* parent);

static void buildOpList(BuildCtx& bc, Reader& r, mlir::Block& block) {
  uint64_t opcnt = r.readULEB();
  for (uint64_t oi = 0; oi < opcnt; ++oi) {
    uint16_t opcode = r.readU16LE();
    uint64_t attrId = r.readULEB();

    if (opcode != kOpcodeGeneric) {
      throw std::runtime_error("Known-op decoding not implemented in ptobc v0 tool (expected generic only)");
    }

    uint64_t nameSid = r.readULEB();
    if (nameSid >= bc.strings->size()) throw std::runtime_error("bad op_name sid");
    std::string opName = (*bc.strings)[nameSid];

    uint64_t nres = r.readULEB();
    llvm::SmallVector<mlir::Type, 4> resTypes;
    resTypes.reserve(nres);

    // Reserve value-id slots for results *before* decoding regions.
    const size_t resStart = bc.values.size();
    for (uint64_t i = 0; i < nres; ++i) {
      uint64_t tid = r.readULEB();
      resTypes.push_back(getType(bc, tid));
      bc.values.push_back(mlir::Value()); // placeholder
    }

    uint64_t nops = r.readULEB();
    llvm::SmallVector<mlir::Value, 8> operands;
    operands.reserve(nops);
    for (uint64_t i = 0; i < nops; ++i) {
      uint64_t vid = r.readULEB();
      if (vid >= bc.values.size()) throw std::runtime_error("operand value_id out of range");
      operands.push_back(bc.values[vid]);
    }

    uint64_t nreg = r.readULEB();

    mlir::OperationState st(mlir::UnknownLoc::get(bc.ctx), opName);
    st.addOperands(operands);
    st.addTypes(resTypes);

    auto dict = getAttrDict(bc, attrId);
    for (auto na : dict) {
      st.addAttribute(na.getName(), na.getValue());
    }

    st.addRegions(nreg);

    mlir::Operation* op = mlir::Operation::create(st);

    // Build regions (preorder)
    for (uint64_t ri = 0; ri < nreg; ++ri) {
      op->getRegion(ri).takeBody(buildRegion(bc, r, op));
    }

    block.getOperations().push_back(op);

    // Fill result value ids (stable even if regions added new values)
    for (uint64_t i = 0; i < nres; ++i) {
      bc.values[resStart + i] = op->getResult(i);
    }
  }
}

static mlir::Region buildRegion(BuildCtx& bc, Reader& r, mlir::Operation* parent) {
  mlir::Region region;
  region.push_back(new mlir::Block());
  uint64_t bcnt = r.readULEB();
  region.getBlocks().clear();
  for (uint64_t bi = 0; bi < bcnt; ++bi) {
    auto* block = new mlir::Block();
    uint64_t nargs = r.readULEB();
    for (uint64_t ai = 0; ai < nargs; ++ai) {
      uint64_t tid = r.readULEB();
      auto ty = getType(bc, tid);
      auto arg = block->addArgument(ty, mlir::UnknownLoc::get(bc.ctx));
      bc.values.push_back(arg);
    }
    buildOpList(bc, r, *block);
    region.push_back(block);
  }
  return region;
}

static mlir::ModuleOp decodeToModule(mlir::MLIRContext& ctx,
                                    const std::vector<std::string>& strings,
                                    const std::vector<TypeEntry>& types,
                                    const std::vector<AttrEntry>& attrs,
                                    const std::vector<uint8_t>& moduleBytes) {
  Reader r{moduleBytes.data(), moduleBytes.data() + moduleBytes.size()};
  uint8_t profile = r.readU8();
  (void)profile;
  uint8_t indexWidth = r.readU8();
  (void)indexWidth;

  uint64_t moduleAttrId = r.readULEB();
  uint64_t gcnt = r.readULEB();
  for (uint64_t i = 0; i < gcnt; ++i) {
    throw std::runtime_error("globals not supported");
  }

  uint64_t fcnt = r.readULEB();

  struct FuncDecl { std::string name; mlir::FunctionType type; mlir::DictionaryAttr attrs; uint8_t flags; };
  std::vector<FuncDecl> decls;
  decls.reserve(fcnt);

  BuildCtx bc{&ctx, &strings, &types, &attrs, {}};

  for (uint64_t i = 0; i < fcnt; ++i) {
    uint64_t nameSid = r.readULEB();
    uint64_t ftypeId = r.readULEB();
    uint8_t flags = r.readU8();
    uint64_t fattrId = r.readULEB();
    if (nameSid >= strings.size()) throw std::runtime_error("bad func name sid");

    auto ty = parseType(ctx, types.at(ftypeId).asmStr);
    auto fty = mlir::dyn_cast<mlir::FunctionType>(ty);
    if (!fty) throw std::runtime_error("func type parse failed");

    decls.push_back({strings[nameSid], fty, getAttrDict(bc, fattrId), flags});
  }

  auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));

  // Apply module attrs
  auto modDict = getAttrDict(bc, moduleAttrId);
  for (auto na : modDict) {
    module->setAttr(na.getName(), na.getValue());
  }

  for (uint64_t i = 0; i < fcnt; ++i) {
    auto fn = mlir::func::FuncOp::create(mlir::UnknownLoc::get(&ctx), decls[i].name, decls[i].type);
    for (auto na : decls[i].attrs) {
      fn->setAttr(na.getName(), na.getValue());
    }

    if ((decls[i].flags & 0x1) == 0) {
      // decode body region
      bc.values.clear();
      mlir::Region body = buildRegion(bc, r, fn);
      fn.getBody().takeBody(body);
    }

    module.push_back(fn);
  }

  if (r.p != r.end) throw std::runtime_error("trailing bytes in MODULE");
  return module;
}

void decodeFileToPTO(const std::string& inPath, const std::string& outPath) {
  auto data = readFile(inPath);
  if (data.size() < 14) throw std::runtime_error("file too small");
  if (std::memcmp(data.data(), "PTOBC\0", 6) != 0) throw std::runtime_error("bad magic");

  uint16_t ver = uint16_t(data[6]) | (uint16_t(data[7]) << 8);
  if (ver != kVersionV0) throw std::runtime_error("unsupported version");

  uint32_t payloadLen = uint32_t(data[10]) | (uint32_t(data[11]) << 8) | (uint32_t(data[12]) << 16) | (uint32_t(data[13]) << 24);
  if (payloadLen != data.size() - 14) throw std::runtime_error("payload_len mismatch");

  Reader r{data.data() + 14, data.data() + data.size()};

  auto readSection = [&]() -> std::pair<uint8_t, std::vector<uint8_t>> {
    uint8_t sid = r.readU8();
    uint32_t slen = r.readU32LE();
    auto bytes = r.readBytes(slen);
    return {sid, bytes};
  };

  auto [s1, d1] = readSection();
  auto [s2, d2] = readSection();
  auto [s3, d3] = readSection();
  auto [s4, d4] = readSection();
  auto [s6, d6] = readSection();

  if (s1 != kSectionStrings || s2 != kSectionTypes || s3 != kSectionAttrs || s4 != kSectionConstPool || s6 != kSectionModule) {
    throw std::runtime_error("unexpected section order");
  }

  std::vector<std::string> strings;
  parseStringsSection(d1, strings);

  std::vector<TypeEntry> types;
  parseTypesSection(d2, strings, types);

  std::vector<AttrEntry> attrs;
  parseAttrsSection(d3, strings, attrs);

  // ignore constpool
  (void)d4;

  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext ctx(registry);
  ctx.allowUnregisteredDialects(true);

  auto module = decodeToModule(ctx, strings, types, attrs, d6);

  std::string out;
  llvm::raw_string_ostream os(out);
  module.print(os);
  os.flush();

  std::ofstream ofs(outPath);
  ofs << out;
}

} // namespace ptobc
