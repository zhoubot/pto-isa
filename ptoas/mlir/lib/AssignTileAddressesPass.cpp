#include "ptoas/Passes.h"
#include "ptoas/ProtoAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
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

static std::string stripIndexing(std::string s) {
  auto l = s.find('[');
  if (l == std::string::npos)
    return trim(s);
  return trim(s.substr(0, l));
}

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

static std::string tileLocOrEmpty(llvm::StringRef typeStr) {
  if (auto loc = findTypeKV(typeStr, "loc="))
    return *loc;
  return "";
}

static bool isDedicatedLoc(llvm::StringRef loc) {
  // These tiles live in dedicated on-core buffers (L0A/L0B/L0C/...) with their own address spaces.
  // We still need distinct offsets when multiple tiles share the same loc (e.g. ping-pong Left/Right tiles).
  return loc == "Left" || loc == "Right" || loc == "Acc" || loc == "Bias" || loc == "ScaleLeft" || loc == "ScaleRight" ||
         loc == "Scaling";
}

static uint64_t stepForLoc(llvm::StringRef loc, llvm::StringRef typeStr) {
  // Empirically align with NPU ST patterns:
  // - Vec tiles can be tightly packed (UB/Vec scratch); keep them 4KB-aligned
  // - Mat tiles use 0x20000 spacing (L1 tiles for cube matmul)
  if (loc == "Vec") {
    auto bytes = tileBytesFromTypeOrDefault(typeStr);
    return alignUp(bytes, 0x1000);
  }
  if (loc == "Mat")
    return 0x20000;
  // L0A/L0B/L0C ping-pong patterns commonly use 32 KiB slots. This is conservative and avoids overlap
  // even when the tile's physical footprint is larger than `fractal`.
  if (isDedicatedLoc(loc))
    return 0x8000;
  // Fallback: keep the old conservative scheme.
  auto bytes = tileBytesFromTypeOrDefault(typeStr);
  return alignUp(bytes, 0x1000);
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
      std::optional<std::string> addrLiteral; // raw literal, if provided
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
        TileDecl d{tileName, typeStr, op, std::nullopt};
        if (operands.size() >= 2 && !trim(operands[1]).empty()) {
          auto addrStr = trim(operands[1]);
          auto addrLit = parseIntLiteralOrZero(addrStr);
          if (addrLit != 0 || addrStr == "0" || addrStr == "0x0" || addrStr == "0X0") {
            d.addrLiteral = std::move(addrStr);
            tilesWithAddr.insert(tileName);
          }
        }
        tiles[tileName] = std::move(d);
        return;
      }
    });

    // Detect split-kernel mode by presence of stage annotations.
    std::vector<std::string> stages;
    std::set<std::string> seenStages;
    module.walk([&](mlir::Operation *op) {
      auto a = op->getAttrOfType<mlir::StringAttr>(kStageAttr);
      if (!a)
        return;
      auto s = trim(a.getValue().str());
      if (s.empty() || s == kPreambleStage)
        return;
      if (seenStages.insert(s).second)
        stages.push_back(std::move(s));
    });

    const bool stageAware = !stages.empty();

    if (stageAware) {
      mlir::Builder b(&getContext());

      // Compute which tiles are used in each stage.
      std::set<std::string> allTileNames;
      for (auto &kv : tiles)
        allTileNames.insert(kv.first);

      std::map<std::string, std::set<std::string>> usedByStage;
      for (auto &s : stages)
        usedByStage[s] = {};

      module.walk([&](mlir::Operation *op) {
        auto name = op->getName().getStringRef();
        if (name == "pto.arg" || name == "pto.const" || name == "pto.make_tensor_view" || name == "pto.alloc_tile")
          return;
        auto ops = readOperands(op);
        if (ops.empty())
          return;
        std::string stage;
        if (auto a = op->getAttrOfType<mlir::StringAttr>(kStageAttr)) {
          stage = trim(a.getValue().str());
          if (stage == kPreambleStage)
            stage.clear();
        }
        for (auto &o : ops) {
          auto base = stripIndexing(o);
          if (!allTileNames.count(base))
            continue;
          if (stage.empty()) {
            // Preamble ops are emitted into every stage kernel.
            for (auto &s : stages)
              usedByStage[s].insert(base);
          } else {
            usedByStage[stage].insert(base);
          }
        }
      });

      // Per-stage allocator state.
      struct AllocState {
        uint64_t nextVec = 0x0;
        uint64_t nextMat = 0x0;
        uint64_t nextOther = 0x10000;
        std::map<std::string, uint64_t> nextDedicated;
      };
      std::map<std::string, AllocState> stateByStage;

      // Seed allocators using any explicit address literals.
      for (auto &stage : stages) {
        auto &st = stateByStage[stage];
        for (auto &tileName : usedByStage[stage]) {
          auto it = tiles.find(tileName);
          if (it == tiles.end())
            continue;
          auto &decl = it->second;
          if (!decl.addrLiteral)
            continue;
          auto addrStr = trim(*decl.addrLiteral);
          auto addrLit = parseIntLiteralOrZero(addrStr);
          if (addrLit == 0 && addrStr != "0" && addrStr != "0x0" && addrStr != "0X0")
            continue;
          auto loc = tileLocOrEmpty(decl.typeStr);
          auto step = stepForLoc(loc, decl.typeStr);
          if (loc == "Vec")
            st.nextVec = std::max(st.nextVec, addrLit + step);
          else if (loc == "Mat")
            st.nextMat = std::max(st.nextMat, addrLit + step);
          else if (isDedicatedLoc(loc))
            st.nextDedicated[loc] = std::max(st.nextDedicated[loc], addrLit + step);
          else
            st.nextOther = std::max(st.nextOther, alignUp(addrLit + step, 0x1000));
        }
      }

      // Helper to allocate one tile address in a stage.
      auto allocOne = [&](const std::string &stage, const TileDecl &decl) -> std::string {
        auto &st = stateByStage[stage];
        auto loc = tileLocOrEmpty(decl.typeStr);
        uint64_t addr = 0;
        if (loc == "Vec") {
          addr = alignUp(st.nextVec, 0x1000);
          st.nextVec = addr + stepForLoc(loc, decl.typeStr);
        } else if (loc == "Mat") {
          addr = alignUp(st.nextMat, 0x20000);
          st.nextMat = addr + stepForLoc(loc, decl.typeStr);
        } else if (isDedicatedLoc(loc)) {
          addr = alignUp(st.nextDedicated[loc], 0x1000);
          st.nextDedicated[loc] = addr + stepForLoc(loc, decl.typeStr);
        } else {
          addr = alignUp(st.nextOther, 0x1000);
          st.nextOther = addr + stepForLoc(loc, decl.typeStr);
        }
        std::ostringstream ss;
        ss << "0x" << std::hex << addr;
        return ss.str();
      };

      // Attach per-stage address maps.
      for (auto &[tileName, decl] : tiles) {
        llvm::SmallVector<mlir::NamedAttribute> kvs;
        kvs.reserve(stages.size());
        for (auto &stage : stages) {
          if (!usedByStage[stage].count(tileName))
            continue;
          std::string addrStr;
          if (decl.addrLiteral) {
            addrStr = *decl.addrLiteral;
          } else {
            addrStr = allocOne(stage, decl);
          }
          kvs.push_back(b.getNamedAttr(stage, b.getStringAttr(addrStr)));
        }
        if (kvs.empty())
          continue;
        decl.declOp->setAttr(kTileAddrMapAttr, b.getDictionaryAttr(kvs));
      }

      return;
    }

    // Track per-loc address ranges.
    uint64_t nextVec = 0x0;
    uint64_t nextMat = 0x0;
    uint64_t nextOther = 0x10000;
    std::map<std::string, uint64_t> nextDedicated;

    for (auto &[tileName, decl] : tiles) {
      if (!tilesWithAddr.count(tileName))
        continue;
      auto operands = readOperands(decl.declOp);
      if (operands.size() < 2)
        continue;
      auto addrLit = parseIntLiteralOrZero(trim(operands[1]));
      if (addrLit == 0 && trim(operands[1]) != "0" && trim(operands[1]) != "0x0" && trim(operands[1]) != "0X0")
        continue;
      auto loc = tileLocOrEmpty(decl.typeStr);
      auto step = stepForLoc(loc, decl.typeStr);
      if (loc == "Vec")
        nextVec = std::max(nextVec, addrLit + step);
      else if (loc == "Mat")
        nextMat = std::max(nextMat, addrLit + step);
      else if (isDedicatedLoc(loc))
        nextDedicated[loc] = std::max(nextDedicated[loc], addrLit + step);
      else
        nextOther = std::max(nextOther, alignUp(addrLit + step, 0x1000));
    }

    // Assign missing addresses (by attaching a numeric literal to `pto.alloc_tile`).
    for (auto &[tileName, decl] : tiles) {
      if (tilesWithAddr.count(tileName))
        continue;
      auto loc = tileLocOrEmpty(decl.typeStr);

      uint64_t addr = 0;
      if (loc == "Vec") {
        addr = alignUp(nextVec, 0x1000);
        nextVec = addr + stepForLoc(loc, decl.typeStr);
      } else if (loc == "Mat") {
        addr = alignUp(nextMat, 0x20000);
        nextMat = addr + stepForLoc(loc, decl.typeStr);
      } else if (isDedicatedLoc(loc)) {
        addr = alignUp(nextDedicated[loc], 0x1000);
        nextDedicated[loc] = addr + stepForLoc(loc, decl.typeStr);
      } else {
        auto step = stepForLoc(loc, decl.typeStr);
        addr = alignUp(nextOther, 0x1000);
        nextOther = addr + step;
      }

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
