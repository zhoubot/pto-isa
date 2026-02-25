#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"

#include "ptobc/leb128.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/SourceMgr.h>

#include <llvm/ADT/DenseMap.h>

#include <cstdlib>
#include <unordered_map>

namespace ptobc {

static uint64_t internType(PTOBCFile& f, mlir::Type t) {
  std::string s = printType(t);
  f.strings.intern(s);
  // type ids are 1-based
  for (size_t i = 0; i < f.typeAsm.size(); ++i) {
    if (f.typeAsm[i] == s) return i + 1;
  }
  f.typeAsm.push_back(s);
  return f.typeAsm.size();
}

static uint64_t internAttr(PTOBCFile& f, mlir::DictionaryAttr dict) {
  if (!dict || dict.empty()) return 0;
  std::string s = printAttrDict(dict);
  f.strings.intern(s);
  for (size_t i = 0; i < f.attrAsm.size(); ++i) {
    if (f.attrAsm[i] == s) return i + 1;
  }
  f.attrAsm.push_back(s);
  return f.attrAsm.size();
}

static std::string hexFloatLiteral(mlir::FloatAttr a) {
  llvm::SmallVector<char, 32> digits;
  llvm::APInt bits = a.getValue().bitcastToAPInt();
  bits.toString(digits, /*Radix=*/16, /*Signed=*/false, /*formatAsCLiteral=*/true);
  return std::string(digits.data(), digits.size());
}

static std::string apIntToSignedDecimal(const llvm::APInt &v) {
  llvm::SmallVector<char, 32> digits;
  v.toString(digits, /*Radix=*/10, /*Signed=*/true, /*formatAsCLiteral=*/false);
  return std::string(digits.data(), digits.size());
}

struct Encoder {
  PTOBCFile file;

  bool emitDebugInfo = false;

  // Per-function numbering state.
  uint64_t funcId = 0;
  uint64_t nextOpId = 0;
  llvm::DenseMap<mlir::Value, uint64_t> valueId;
  std::vector<mlir::Value> valueById;

  // Module-wide debug file table state.
  std::unordered_map<std::string, uint64_t> dbgFileIdByPath;

  uint64_t getValueId(mlir::Value v) {
    auto it = valueId.find(v);
    if (it == valueId.end()) {
      throw std::runtime_error("operand references undefined value");
    }
    return it->second;
  }

  uint64_t allocValueId(mlir::Value v) {
    uint64_t id = valueId.size();
    auto [it, inserted] = valueId.try_emplace(v, id);
    if (!inserted) throw std::runtime_error("value already has an id");
    valueById.push_back(v);
    return it->second;
  }

  uint64_t internDbgFile(llvm::StringRef path) {
    auto p = path.str();
    auto it = dbgFileIdByPath.find(p);
    if (it != dbgFileIdByPath.end()) return it->second;

    uint64_t sid = file.strings.intern(p);
    uint64_t fileId = file.dbgFiles.size();
    file.dbgFiles.push_back(DebugFileEntry{sid, /*hashKind=*/0, {}});
    dbgFileIdByPath.emplace(std::move(p), fileId);
    return fileId;
  }

  void recordOpLocation(uint64_t opId, mlir::Operation &op) {
    if (!emitDebugInfo) return;
    auto loc = op.getLoc();
    auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(loc);
    if (!flc) return;

    uint64_t fileId = internDbgFile(flc.getFilename().getValue());
    uint64_t sl = flc.getLine();
    uint64_t sc = flc.getColumn();
    uint64_t el = sl;
    uint64_t ec = sc + 1; // point-range

    file.dbgLocations.push_back(DebugLocationEntry{funcId, opId, fileId, sl, sc, el, ec});
  }

  void finalizeValueNamesForFunction() {
    if (!emitDebugInfo) return;
    // Deterministic value names for DebugInfo.
    std::unordered_map<std::string, int> constCounts;

    for (uint64_t vid = 0; vid < valueById.size(); ++vid) {
      mlir::Value v = valueById[vid];
      std::string name;

      if (auto *def = v.getDefiningOp()) {
        if (auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(def)) {
          mlir::Attribute a = cst.getValue();
          std::string ty = printType(v.getType());

          // Only generate special names for scalar ints/floats.
          if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(a)) {
            std::string imm = hexFloatLiteral(fa);
            std::string base = "c" + imm + "_" + ty;
            int &n = constCounts[base];
            name = base;
            if (n > 0) name += "_" + std::to_string(n);
            ++n;
          } else if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(a)) {
            std::string imm = apIntToSignedDecimal(ia.getValue());
            std::string base = "c" + imm;
            if (ty != "index") base += "_" + ty;
            int &n = constCounts[base];
            name = base;
            if (n > 0) name += "_" + std::to_string(n);
            ++n;
          }
        }
      }

      if (name.empty()) {
        // Non-constant (or non-scalar-constant) value.
        name = std::to_string(vid);
      }

      uint64_t nameSid = file.strings.intern(name);
      file.dbgValueNames.push_back(DebugValueNameEntry{funcId, vid, nameSid});
    }
  }

  void resetForFunction(uint64_t fid) {
    funcId = fid;
    nextOpId = 0;
    valueId.clear();
    valueById.clear();
  }

  void encodeRegion(mlir::Region& region, Buffer& out);
  void encodeBlock(mlir::Block& block, Buffer& out);
  void encodeOp(mlir::Operation& op, Buffer& out);
};

void Encoder::encodeRegion(mlir::Region& region, Buffer& out) {
  writeULEB128(region.getBlocks().size(), out.bytes);
  for (auto& block : region.getBlocks()) {
    encodeBlock(block, out);
  }
}

void Encoder::encodeBlock(mlir::Block& block, Buffer& out) {
  // block args
  writeULEB128(block.getNumArguments(), out.bytes);
  for (auto arg : block.getArguments()) {
    writeULEB128(internType(file, arg.getType()), out.bytes);
    allocValueId(arg);
  }

  // ops count
  size_t opCount = 0;
  for (auto& op : block.getOperations()) (void)op, ++opCount;
  writeULEB128(opCount, out.bytes);

  for (auto& op : block.getOperations()) {
    encodeOp(op, out);
  }
}

void Encoder::encodeOp(mlir::Operation& op, Buffer& out) {
  if (emitDebugInfo) {
    // op_id (preorder DFS, per-function)
    uint64_t opId = nextOpId++;
    recordOpLocation(opId, op);
  }

  // opcode (generic)
  out.appendU16LE(kOpcodeGeneric);

  // attr_id
  auto attrId = internAttr(file, op.getAttrDictionary());
  writeULEB128(attrId, out.bytes);

  // op-name
  auto opName = op.getName().getStringRef().str();
  auto opNameSid = file.strings.intern(opName);
  writeULEB128(opNameSid, out.bytes);

  // results
  writeULEB128(op.getNumResults(), out.bytes);
  for (auto res : op.getResults()) {
    // allocate id now (preorder semantics)
    allocValueId(res);
    writeULEB128(internType(file, res.getType()), out.bytes);
  }

  // operands
  writeULEB128(op.getNumOperands(), out.bytes);
  for (auto operand : op.getOperands()) {
    writeULEB128(getValueId(operand), out.bytes);
  }

  // regions
  writeULEB128(op.getNumRegions(), out.bytes);
  for (auto& r : op.getRegions()) {
    encodeRegion(r, out);
  }
}

PTOBCFile encodeFromMLIRModule(mlir::ModuleOp module) {
  Encoder enc;
  enc.emitDebugInfo = (std::getenv("PTOBC_EMIT_DEBUGINFO") != nullptr);

  // Pre-intern a few common strings to stabilize ids.
  enc.file.strings.intern("func.func");
  enc.file.strings.intern("func.return");

  // MODULE encoding
  Buffer m;
  // profile_id=0 (unspecified), index_width=64
  m.appendU8(0);
  m.appendU8(64);

  // module_attr_id
  uint64_t modAttrId = internAttr(enc.file, module->getAttrDictionary());
  writeULEB128(modAttrId, m.bytes);

  // globals count
  writeULEB128(0, m.bytes);

  // function decls (top-level order)
  llvm::SmallVector<mlir::func::FuncOp, 8> funcs;
  for (auto f : module.getOps<mlir::func::FuncOp>()) {
    funcs.push_back(f);
  }

  writeULEB128(funcs.size(), m.bytes);

  // encode decls
  for (auto f : funcs) {
    auto nameSid = enc.file.strings.intern(f.getName().str());
    // func type as opaque asm in type table
    auto funcTypeId = internType(enc.file, f.getFunctionType());
    // flags: bit0 import? (0)
    uint8_t flags = 0;
    auto funcAttrId = internAttr(enc.file, f->getAttrDictionary());

    writeULEB128(nameSid, m.bytes);
    writeULEB128(funcTypeId, m.bytes);
    m.appendU8(flags);
    writeULEB128(funcAttrId, m.bytes);
  }

  // bodies: for each function, encode its body region
  for (size_t i = 0; i < funcs.size(); ++i) {
    auto f = funcs[i];
    enc.resetForFunction(i);

    // function body is region #0
    enc.encodeRegion(f.getBody(), m);

    // DebugInfo: deterministic value names for this function.
    enc.finalizeValueNamesForFunction();
  }

  enc.file.moduleBytes = std::move(m.bytes);
  return enc.file;
}

mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path) {
  llvm::SourceMgr sm;
  std::string err;
  auto file = mlir::openInputFile(path, &err);
  if (!file) {
    throw std::runtime_error("failed to open input: " + path + (err.empty() ? "" : (": " + err)));
  }
  sm.AddNewSourceBuffer(std::move(file), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);
  if (!module) {
    throw std::runtime_error("failed to parse MLIR file: " + path);
  }
  return module;
}

} // namespace ptobc
