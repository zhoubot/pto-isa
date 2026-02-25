#include "ptobc/mlir_helpers.h"
#include "ptobc/ptobc_format.h"

#include "ptobc/leb128.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/LogicalResult.h>

#include <llvm/ADT/StringRef.h>
#include <llvm/Support/SourceMgr.h>

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

struct Encoder {
  PTOBCFile file;
  std::unordered_map<mlir::Value, uint64_t> valueId;

  uint64_t getValueId(mlir::Value v) {
    auto it = valueId.find(v);
    if (it == valueId.end()) {
      throw std::runtime_error("operand references undefined value");
    }
    return it->second;
  }

  uint64_t allocValueId(mlir::Value v) {
    uint64_t id = valueId.size();
    valueId.emplace(v, id);
    return id;
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

  // function decls
  llvm::SmallVector<mlir::func::FuncOp, 8> funcs;
  module.walk([&](mlir::func::FuncOp f) { funcs.push_back(f); });

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
  for (auto f : funcs) {
    enc.valueId.clear();
    // function body is region #0
    enc.encodeRegion(f.getBody(), m);
  }

  enc.file.moduleBytes = std::move(m.bytes);
  return enc.file;
}

mlir::OwningOpRef<mlir::ModuleOp> parsePTOFile(mlir::MLIRContext& ctx, const std::string& path) {
  llvm::SourceMgr sm;
  auto fileOrErr = mlir::openInputFile(path);
  if (!fileOrErr) {
    throw std::runtime_error("failed to open input: " + path);
  }
  sm.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sm, &ctx);
  if (!module) {
    throw std::runtime_error("failed to parse MLIR file: " + path);
  }
  return module;
}

} // namespace ptobc
