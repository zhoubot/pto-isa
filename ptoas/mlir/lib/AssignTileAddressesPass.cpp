#include "ptoas/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace ptoas {
namespace {

static std::string trim(std::string s) {
  auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && isSpace((unsigned char)s.front()))
    s.erase(s.begin());
  while (!s.empty() && isSpace((unsigned char)s.back()))
    s.pop_back();
  return s;
}

static std::vector<std::string> readOperands(mlir::Operation *op) {
  auto arr = op->getAttrOfType<mlir::ArrayAttr>("operands");
  if (!arr)
    return {};
  std::vector<std::string> out;
  out.reserve(arr.size());
  for (auto a : arr) {
    auto s = llvm::dyn_cast<mlir::StringAttr>(a);
    if (!s)
      llvm::report_fatal_error("operands must be string attrs");
    out.push_back(s.getValue().str());
  }
  return out;
}

static bool isTileTypeString(llvm::StringRef typeStr) { return typeStr.starts_with("!pto.tile"); }

static std::optional<std::string> findTypeKV(llvm::StringRef typeStr, llvm::StringRef key) {
  auto p = typeStr.find(key);
  if (p == llvm::StringRef::npos)
    return std::nullopt;
  p += key.size();
  auto end = typeStr.find_first_of(",>", p);
  if (end == llvm::StringRef::npos)
    end = typeStr.size();
  return typeStr.slice(p, end).trim().str();
}

static uint64_t parseIntLiteralOrZero(const std::string &s) {
  auto t = trim(s);
  if (t.rfind("0x", 0) == 0 || t.rfind("0X", 0) == 0) {
    char *end = nullptr;
    auto v = std::strtoull(t.c_str(), &end, 16);
    if (end && *end == '\0')
      return v;
    return 0;
  }
  char *end = nullptr;
  auto v = std::strtoull(t.c_str(), &end, 10);
  if (end && *end == '\0')
    return v;
  return 0;
}

static uint64_t alignUp(uint64_t v, uint64_t align) {
  if (align == 0)
    return v;
  auto mask = align - 1;
  return (v + mask) & ~mask;
}

static uint64_t tileBytesFromTypeOrDefault(llvm::StringRef typeStr) {
  // Prefer `fractal=<bytes>` when present, else fall back to a conservative default.
  if (auto fractal = findTypeKV(typeStr, "fractal=")) {
    auto bytes = parseIntLiteralOrZero(*fractal);
    if (bytes != 0)
      return bytes;
  }
  if (auto loc = findTypeKV(typeStr, "loc=")) {
    if (*loc == "Acc")
      return 1024;
  }
  return 512;
}

struct AssignTileAddressesPass
    : public mlir::PassWrapper<AssignTileAddressesPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *top = module.getBody();
    if (!top)
      return;

    // Collect tile declarations and existing address bindings.
    struct TileDecl {
      std::string name;
      std::string typeStr;
      mlir::Operation *declOp = nullptr; // pto.alloc_tile
    };
    std::map<std::string, TileDecl> tiles;
    std::set<std::string> tilesWithAddr;

    module.walk([&](mlir::Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "pto.alloc_tile") {
        auto operands = readOperands(op);
        if (operands.empty())
          return;
        auto tileName = trim(operands[0]);
        auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig");
        if (!typeSig)
          return;
        auto typeStr = trim(typeSig.getValue().str());
        if (!isTileTypeString(typeStr))
          return;
        tiles[tileName] = TileDecl{tileName, typeStr, op};
        if (operands.size() >= 2 && !trim(operands[1]).empty())
          tilesWithAddr.insert(tileName);
        return;
      }
    });

    // Compute starting address after any existing literal address assignments we can resolve.
    uint64_t nextAddr = 0x10000;
    for (auto &[tileName, decl] : tiles) {
      if (!tilesWithAddr.count(tileName))
        continue;
      auto operands = readOperands(decl.declOp);
      if (operands.size() < 2)
        continue;
      auto addrLit = parseIntLiteralOrZero(trim(operands[1]));
      if (addrLit == 0 && trim(operands[1]) != "0" && trim(operands[1]) != "0x0" && trim(operands[1]) != "0X0")
        continue;
      auto bytes = tileBytesFromTypeOrDefault(decl.typeStr);
      auto step = alignUp(bytes, 0x1000);
      nextAddr = std::max(nextAddr, alignUp(addrLit + step, 0x1000));
    }

    // Assign missing addresses (by attaching a numeric literal to `pto.alloc_tile`).
    for (auto &[tileName, decl] : tiles) {
      if (tilesWithAddr.count(tileName))
        continue;
      auto bytes = tileBytesFromTypeOrDefault(decl.typeStr);
      auto step = alignUp(bytes, 0x1000);
      auto addr = alignUp(nextAddr, 0x1000);
      nextAddr = addr + step;

      std::ostringstream ss;
      ss << "0x" << std::hex << addr;
      auto addrLit = ss.str();

      mlir::OpBuilder b(module.getContext());
      auto operands = readOperands(decl.declOp);
      llvm::SmallVector<mlir::Attribute> attrs;
      attrs.push_back(b.getStringAttr(decl.name));
      attrs.push_back(b.getStringAttr(addrLit));
      // Preserve any extra operands (if present).
      for (size_t i = 1; i < operands.size(); ++i)
        attrs.push_back(b.getStringAttr(operands[i]));
      decl.declOp->setAttr("operands", b.getArrayAttr(attrs));
      tilesWithAddr.insert(tileName);
    }
  }

  llvm::StringRef getArgument() const final { return "ptoas-assign-tile-addrs"; }
  llvm::StringRef getDescription() const final { return "Assign default addresses to tile locals (prototype)."; }
};

} // namespace

std::unique_ptr<mlir::Pass> createAssignTileAddressesPass() { return std::make_unique<AssignTileAddressesPass>(); }

} // namespace ptoas
