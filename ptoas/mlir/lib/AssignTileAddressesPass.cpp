#include "ptoas/Passes.h"
#include "ptoas/ProtoAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <queue>
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

static uint64_t avoidReservedVecUbRange(uint64_t addr, uint64_t step) {
  // A2/A3 UB scratch reserves [TMP_UB_OFFSET, TMP_UB_OFFSET + TMP_UB_SIZE) for compiler helpers.
  // Keep auto-assigned Vec tile addresses out of that range to avoid clobbering scratch.
  constexpr uint64_t kTmpUbOffset = 184ull * 1024ull;
  constexpr uint64_t kTmpUbSize = 8ull * 1024ull;
  constexpr uint64_t reservedStart = kTmpUbOffset;
  constexpr uint64_t reservedEnd = kTmpUbOffset + kTmpUbSize;

  // Addresses/steps are 4KB-aligned, so overlap checks are cheap.
  const uint64_t end = addr + step;
  if (addr < reservedStart && end > reservedStart)
    return alignUp(reservedEnd, 0x1000);
  if (addr >= reservedStart && addr < reservedEnd)
    return alignUp(reservedEnd, 0x1000);
  return addr;
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
    const bool isRegisterBase = [&]() -> bool {
      auto mm = module->getAttrOfType<mlir::StringAttr>(kMemoryModelAttr);
      if (!mm)
        return false;
      return trim(mm.getValue().str()) == "REGISTER_BASE";
    }();

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

      if (isRegisterBase) {
        // REGISTER_BASE (A5) uses banked on-core tile buffers:
        // - Vec: 192KB (3 x 64KB banks)
        // - Mat: 512KB (8 x 64KB banks)
        //
        // Assigning dense byte offsets (e.g. 0x1000) can trigger illegal-instruction
        // faults. Instead, do a small liveness-based bank allocation per stage.
        struct Interval {
          uint32_t start = std::numeric_limits<uint32_t>::max();
          uint32_t end = 0;
        };

        // Build per-stage linear op indices (preamble ops count for all stages).
        std::map<std::string, uint32_t> nextIdx;
        std::map<std::string, std::map<std::string, Interval>> intervals;
        for (auto &s : stages)
          nextIdx[s] = 0;

        auto recordUse = [&](const std::string &stage, const std::string &tileName, uint32_t idx) {
          auto &iv = intervals[stage][tileName];
          iv.start = std::min(iv.start, idx);
          iv.end = std::max(iv.end, idx);
        };

        std::function<void(mlir::Block &)> visitBlock = [&](mlir::Block &blk) {
          for (auto &opRef : blk) {
            auto *op = &opRef;
            auto opname = op->getName().getStringRef();

            const bool isDecl = opname == "pto.arg" || opname == "pto.const" || opname == "pto.make_tensor_view" ||
                                opname == "pto.alloc_tile";
            const bool isPtoOp = opname.starts_with("pto.") && !isDecl;

            std::string stage;
            if (auto a = op->getAttrOfType<mlir::StringAttr>(kStageAttr)) {
              stage = trim(a.getValue().str());
              if (stage == kPreambleStage)
                stage.clear();
            }

            if (isPtoOp) {
              auto ops = readOperands(op);
              if (!ops.empty()) {
                if (stage.empty()) {
                  for (auto &s : stages) {
                    uint32_t idx = nextIdx[s]++;
                    for (auto &o : ops) {
                      auto base = stripIndexing(o);
                      if (!usedByStage[s].count(base))
                        continue;
                      recordUse(s, base, idx);
                    }
                  }
                } else if (usedByStage.count(stage)) {
                  uint32_t idx = nextIdx[stage]++;
                  for (auto &o : ops) {
                    auto base = stripIndexing(o);
                    if (!usedByStage[stage].count(base))
                      continue;
                    recordUse(stage, base, idx);
                  }
                }
              } else {
                // Still advance indices so later ops have stable relative ordering.
                if (stage.empty()) {
                  for (auto &s : stages)
                    nextIdx[s]++;
                } else if (usedByStage.count(stage)) {
                  nextIdx[stage]++;
                }
              }
            }

            for (auto &r : op->getRegions()) {
              if (r.empty())
                continue;
              visitBlock(r.front());
            }
          }
        };

        visitBlock(*top);

        struct BankCfg {
          uint64_t base = 0;
          uint64_t slotBytes = 0;
          uint32_t slots = 0;
        };
        auto cfgForLoc = [&](llvm::StringRef loc) -> std::optional<BankCfg> {
          // Hardware limits (REGISTER_BASE):
          // - Vec tiles: 192KB
          // - Mat tiles: 512KB
          //
          // Use 4KB slots to keep addresses aligned while allowing dense packing.
          // If a future target requires coarser-grained banking, tighten these slots and
          // insert real spills to GM instead of failing.
          // Vec (UB): allow dense packing; 4KB alignment is conservative and matches ST patterns.
          if (loc == "Vec")
            return BankCfg{0x0, 0x1000, 0x30000 / 0x1000};
          // Mat (CBUF/L1): ST patterns and toolchain behavior indicate 64KB granularity.
          if (loc == "Mat")
            return BankCfg{0x0, 0x10000, 0x80000 / 0x10000};
          return std::nullopt;
        };

        // Stage -> tile -> assigned addr.
        std::map<std::string, std::map<std::string, std::string>> addrByStage;

        // Heuristic ordering for Mat tiles on A5:
        // Many ST kernels consistently assign:
        //   - A-matrix Mat tile (fed into Left)  -> 0x00000
        //   - B-matrix Mat tile (fed into Right)-> 0x10000
        //   - Bias Mat tile (fed into Bias)     -> 0x20000
        //
        // Some A5 ops appear to assume these conventional base addresses for their source Mat tiles.
        // Sorting purely by tile name can swap A/B (e.g. kt_mat < q_mat) and lead to wrong results or faults.
        std::map<std::string, std::map<std::string, int>> matPriorityByStage;
        for (auto &s : stages)
          matPriorityByStage[s] = {};
        auto bumpMatPriority = [&](const std::string &stage, const std::string &matName, int pri) {
          if (!usedByStage[stage].count(matName))
            return;
          auto &m = matPriorityByStage[stage];
          auto it = m.find(matName);
          if (it == m.end())
            m[matName] = pri;
          else
            it->second = std::min(it->second, pri);
        };
        module.walk([&](mlir::Operation *op) {
          auto name = op->getName().getStringRef();
          if (name != "pto.tmov")
            return;
          auto ops = readOperands(op);
          if (ops.size() != 2)
            return;
          auto dstName = stripIndexing(ops[0]);
          auto srcName = stripIndexing(ops[1]);
          auto srcIt = tiles.find(srcName);
          auto dstIt = tiles.find(dstName);
          if (srcIt == tiles.end() || dstIt == tiles.end())
            return;
          auto srcLoc = tileLocOrEmpty(srcIt->second.typeStr);
          auto dstLoc = tileLocOrEmpty(dstIt->second.typeStr);
          if (srcLoc != "Mat")
            return;

          int pri = 100;
          if (dstLoc == "Left")
            pri = 0;
          else if (dstLoc == "Right")
            pri = 1;
          else if (dstLoc == "Bias")
            pri = 2;

          std::string stage;
          if (auto a = op->getAttrOfType<mlir::StringAttr>(kStageAttr)) {
            stage = trim(a.getValue().str());
            if (stage == kPreambleStage)
              stage.clear();
          }
          if (stage.empty()) {
            // Preamble ops are emitted into every stage kernel.
            for (auto &s : stages)
              bumpMatPriority(s, srcName, pri);
          } else if (usedByStage.count(stage)) {
            bumpMatPriority(stage, srcName, pri);
          }
        });

        // Allocate banked locs (Vec/Mat) per stage.
        //
        // NOTE: We intentionally do *not* reuse slots based on liveness here.
        // The IR contains loops/ifs, and this prototype pass does not compute
        // accurate liveness across control flow; reusing slots can cause
        // overlapping tile buffers (e.g. clobbering a loop-carried accumulator),
        // which may manifest as wrong results or device faults.
        for (auto &stageName : stages) {
          // Group tiles by loc.
          std::map<std::string, std::vector<std::string>> tilesByLoc;
          for (auto &tileName : usedByStage[stageName]) {
            auto it = tiles.find(tileName);
            if (it == tiles.end())
              continue;
            auto loc = tileLocOrEmpty(it->second.typeStr);
            if (cfgForLoc(loc))
              tilesByLoc[loc].push_back(tileName);
          }

          for (auto &[loc, names] : tilesByLoc) {
            auto cfgOpt = cfgForLoc(loc);
            if (!cfgOpt)
              continue;
            auto cfg = *cfgOpt;

            struct Item {
              std::string name;
              std::optional<uint32_t> fixedSlot;
            };
            std::vector<Item> items;
            items.reserve(names.size());

            for (auto &tname : names) {
              Item it;
              it.name = tname;
              auto &decl = tiles.at(tname);
              if (decl.addrLiteral) {
                auto addrLit = parseIntLiteralOrZero(*decl.addrLiteral);
                if (cfg.slotBytes && (addrLit % cfg.slotBytes == 0)) {
                  uint32_t slot = static_cast<uint32_t>(addrLit / cfg.slotBytes);
                  if (slot < cfg.slots)
                    it.fixedSlot = slot;
                }
              }
              items.push_back(std::move(it));
            }

            if (loc == "Mat") {
              auto &prio = matPriorityByStage[stageName];
              std::sort(items.begin(), items.end(), [&](const Item &a, const Item &b) {
                int pa = 100;
                int pb = 100;
                if (auto ita = prio.find(a.name); ita != prio.end())
                  pa = ita->second;
                if (auto itb = prio.find(b.name); itb != prio.end())
                  pb = itb->second;
                if (pa != pb)
                  return pa < pb;
                return a.name < b.name;
              });
            } else {
              std::sort(items.begin(), items.end(), [&](const Item &a, const Item &b) { return a.name < b.name; });
            }

            std::vector<bool> used(cfg.slots, false);

            for (auto &it : items) {
              uint32_t slot = std::numeric_limits<uint32_t>::max();
              if (it.fixedSlot) {
                slot = *it.fixedSlot;
                if (slot >= cfg.slots)
                  llvm::report_fatal_error("fixedSlot out of range");
                if (used[slot]) {
                  llvm::report_fatal_error(llvm::Twine("REGISTER_BASE tile bank conflict (fixed addr) in stage ") +
                                           stageName + " loc=" + loc + " tile=" + it.name);
                }
              } else {
                for (uint32_t s = 0; s < cfg.slots; ++s) {
                  if (!used[s]) {
                    slot = s;
                    break;
                  }
                }
                if (slot == std::numeric_limits<uint32_t>::max()) {
                  llvm::report_fatal_error(llvm::Twine("REGISTER_BASE out of tile banks in stage ") + stageName +
                                           " loc=" + loc + " (need spill/rewrite): tile=" + it.name);
                }
              }

              used[slot] = true;

              uint64_t addr = cfg.base + static_cast<uint64_t>(slot) * cfg.slotBytes;
              std::ostringstream ss;
              ss << "0x" << std::hex << addr;
              addrByStage[stageName][it.name] = ss.str();
            }
          }
        }

        // Attach per-stage address maps:
        // - Vec/Mat: banked mapping from addrByStage
        // - Other locs: keep the old sequential scheme (per-stage) as a fallback
        std::map<std::string, std::map<std::string, uint32_t>> dedicatedCountByStage;
        for (auto &stageName : stages) {
          for (auto &tileName : usedByStage[stageName]) {
            auto it = tiles.find(tileName);
            if (it == tiles.end())
              continue;
            auto loc = tileLocOrEmpty(it->second.typeStr);
            if (isDedicatedLoc(loc))
              dedicatedCountByStage[stageName][loc]++;
          }
        }

        struct AllocState {
          uint64_t nextOther = 0x10000;
          std::map<std::string, uint64_t> nextDedicated;
        };
        std::map<std::string, AllocState> stateByStage;

        auto allocFallback = [&](const std::string &stage, const TileDecl &decl) -> std::string {
          auto &st = stateByStage[stage];
          auto loc = tileLocOrEmpty(decl.typeStr);
          uint64_t addr = 0;
          if (isDedicatedLoc(loc)) {
            // Under REGISTER_BASE, L0A/L0B/L0C/... tiles are in dedicated address spaces.
            // Most ST kernels bind the single tile per loc to 0x0. However, real kernels
            // often need ping-pong (multiple tiles per loc); allocate distinct slots in
            // that case to avoid clobbering in-flight tiles.
            if (dedicatedCountByStage[stage][loc] <= 1) {
              addr = 0x0;
            } else {
              auto &next = st.nextDedicated[loc];
              addr = alignUp(next, 0x8000);
              next = addr + stepForLoc(loc, decl.typeStr);
            }
          } else {
            addr = alignUp(st.nextOther, 0x1000);
            st.nextOther = addr + stepForLoc(loc, decl.typeStr);
          }
          std::ostringstream ss;
          ss << "0x" << std::hex << addr;
          return ss.str();
        };

        for (auto &[tileName, decl] : tiles) {
          llvm::SmallVector<mlir::NamedAttribute> kvs;
          kvs.reserve(stages.size());
          for (auto &stage : stages) {
            if (!usedByStage[stage].count(tileName))
              continue;
            std::string addrStr;
            if (decl.addrLiteral) {
              addrStr = *decl.addrLiteral;
            } else if (auto it = addrByStage[stage].find(tileName); it != addrByStage[stage].end()) {
              addrStr = it->second;
            } else {
              addrStr = allocFallback(stage, decl);
            }
            kvs.push_back(b.getNamedAttr(stage, b.getStringAttr(addrStr)));
          }
          if (kvs.empty())
            continue;
          decl.declOp->setAttr(kTileAddrMapAttr, b.getDictionaryAttr(kvs));
        }

        return;
      }

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
        const uint64_t step = stepForLoc(loc, decl.typeStr);
        uint64_t addr = 0;
        if (loc == "Vec") {
          addr = alignUp(st.nextVec, 0x1000);
          addr = avoidReservedVecUbRange(addr, step);
          st.nextVec = addr + step;
        } else if (loc == "Mat") {
          addr = alignUp(st.nextMat, 0x20000);
          st.nextMat = addr + step;
        } else if (isDedicatedLoc(loc)) {
          addr = alignUp(st.nextDedicated[loc], 0x1000);
          st.nextDedicated[loc] = addr + step;
        } else {
          addr = alignUp(st.nextOther, 0x1000);
          st.nextOther = addr + step;
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
      else if (isDedicatedLoc(loc))
        nextDedicated[loc] = std::max(nextDedicated[loc], addrLit + step);
      else
        nextOther = std::max(nextOther, alignUp(addrLit + step, 0x1000));
    }

    // Hardware scratch size limits for MEMORY_BASE (A2/A3):
    // - Vec(UB): 192KB total, but [184KB,192KB) is reserved for compiler helpers.
    // - Mat(L1): 512KB total.
    // - Left/Right: 64KB each.
    // - Acc: 128KB total.
    //
    // The prototype frontend declares many tiles up front. To avoid overflowing these fixed buffers,
    // perform a simple live-range based address reuse for Mat and dedicated L0 tiles.
    struct LiveRange {
      std::string name;
      int first = std::numeric_limits<int>::max();
      int last = std::numeric_limits<int>::min();
    };

    std::map<std::string, LiveRange> ranges;
    for (auto &[tileName, decl] : tiles) {
      (void)decl;
      ranges[tileName] = LiveRange{tileName};
    }

    auto recordUseAtIndex = [&](llvm::StringRef tileName, int idx) -> void {
      auto it = ranges.find(tileName.str());
      if (it == ranges.end())
        return;
      it->second.first = std::min(it->second.first, idx);
      it->second.last = std::max(it->second.last, idx);
    };

    int topIndex = 0;
    for (auto &topOp : *top) {
      std::set<std::string> usedHere;
      topOp.walk([&](mlir::Operation *op) {
        auto name = op->getName().getStringRef();
        if (name == "pto.alloc_tile")
          return;
        auto operands = readOperands(op);
        for (auto &o : operands) {
          auto base = stripIndexing(trim(o));
          if (tiles.count(base))
            usedHere.insert(base);
        }
      });
      for (auto &t : usedHere)
        recordUseAtIndex(t, topIndex);
      topIndex++;
    }

    // Build a per-tile address plan for Mat and dedicated locs.
    std::map<std::string, std::string> plannedAddr;

    auto allocWithReuse = [&](llvm::StringRef loc, uint64_t totalBytes, uint64_t slotBytes) -> void {
      if (slotBytes == 0)
        return;
      const uint64_t maxSlots = totalBytes / slotBytes;
      if (maxSlots == 0)
        return;

      struct Item {
        std::string name;
        int first;
        int last;
      };
      std::vector<Item> items;
      for (auto &[tileName, decl] : tiles) {
        if (tilesWithAddr.count(tileName))
          continue;
        if (tileLocOrEmpty(decl.typeStr) != loc)
          continue;
        auto it = ranges.find(tileName);
        int first = std::numeric_limits<int>::max();
        int last = std::numeric_limits<int>::min();
        if (it != ranges.end()) {
          first = it->second.first;
          last = it->second.last;
        }
        // Unused tiles: place them at the end to maximize reuse.
        if (first == std::numeric_limits<int>::max() || last == std::numeric_limits<int>::min()) {
          first = std::numeric_limits<int>::max() / 2;
          last = first;
        }
        items.push_back(Item{tileName, first, last});
      }
      if (items.empty())
        return;

      std::sort(items.begin(), items.end(), [&](const Item &a, const Item &b) {
        if (a.first != b.first)
          return a.first < b.first;
        return a.last < b.last;
      });

      // Reuse slots based on conservative live ranges. This keeps addresses within fixed on-core banks
      // (Vec/Mat/L0*) while avoiding overlap for simultaneously-live tiles.
      struct Active {
        int last;
        uint64_t slot;
      };
      auto byLast = [](const Active &a, const Active &b) { return a.last > b.last; };
      std::priority_queue<Active, std::vector<Active>, decltype(byLast)> active(byLast);

      auto bySlot = [](uint64_t a, uint64_t b) { return a > b; };
      std::priority_queue<uint64_t, std::vector<uint64_t>, decltype(bySlot)> freeSlots(bySlot);

      uint64_t nextSlot = 0;
      for (auto &it : items) {
        while (!active.empty() && active.top().last < it.first) {
          freeSlots.push(active.top().slot);
          active.pop();
        }

        uint64_t slot = 0;
        if (!freeSlots.empty()) {
          slot = freeSlots.top();
          freeSlots.pop();
        } else {
          if (nextSlot >= maxSlots) {
            llvm::report_fatal_error(llvm::Twine("out of tile banks (need spill/rewrite): loc=") + loc +
                                     " (slots=" + std::to_string(maxSlots) + ") tile=" + it.name);
          }
          slot = nextSlot++;
        }

        uint64_t addr = slot * slotBytes;
        std::ostringstream ss;
        ss << "0x" << std::hex << addr;
        plannedAddr[it.name] = ss.str();
        active.push(Active{it.last, slot});
      }
    };

    // Mat(L1): 512KB total, allocate in 0x20000 slots (matches common L1 bank spacing on A2/A3).
    allocWithReuse("Mat", 512ull * 1024ull, 0x20000);
    // Dedicated locs: allocate within their fixed banks.
    allocWithReuse("Left", 64ull * 1024ull, 0x8000);
    allocWithReuse("Right", 64ull * 1024ull, 0x8000);
    allocWithReuse("Acc", 128ull * 1024ull, 0x8000);
    allocWithReuse("Bias", 64ull * 1024ull, 0x8000);
    allocWithReuse("ScaleLeft", 64ull * 1024ull, 0x8000);
    allocWithReuse("ScaleRight", 64ull * 1024ull, 0x8000);
    allocWithReuse("Scaling", 64ull * 1024ull, 0x8000);

    // Assign missing addresses (by attaching a numeric literal to `pto.alloc_tile`).
    for (auto &[tileName, decl] : tiles) {
      if (tilesWithAddr.count(tileName))
        continue;
      auto loc = tileLocOrEmpty(decl.typeStr);

      uint64_t addr = 0;
      std::string addrLit;
      if (auto it = plannedAddr.find(tileName); it != plannedAddr.end()) {
        addrLit = it->second;
        addr = parseIntLiteralOrZero(addrLit);
      } else if (loc == "Vec") {
        addr = alignUp(nextVec, 0x1000);
        auto step = stepForLoc(loc, decl.typeStr);
        addr = avoidReservedVecUbRange(addr, step);
        nextVec = addr + step;
      } else if (isDedicatedLoc(loc)) {
        // Fallback for dedicated locs not in our fixed list.
        addr = alignUp(nextDedicated[loc], 0x1000);
        nextDedicated[loc] = addr + stepForLoc(loc, decl.typeStr);
      } else {
        auto step = stepForLoc(loc, decl.typeStr);
        addr = alignUp(nextOther, 0x1000);
        nextOther = addr + step;
      }
      if (addrLit.empty()) {
        std::ostringstream ss;
        ss << "0x" << std::hex << addr;
        addrLit = ss.str();
      }

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
