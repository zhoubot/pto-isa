#include "ptoas/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <string>
#include <tuple>
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

static llvm::StringRef stripDialect(llvm::StringRef opName) {
  if (auto dot = opName.find('.'); dot != llvm::StringRef::npos)
    return opName.drop_front(dot + 1);
  return opName;
}

static bool isPtoMetaOp(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "pto.arg" || name == "pto.const" || name == "pto.make_tensor_view" || name == "pto.subview" ||
         name == "pto.alloc_tile" || name == "pto.record_event" || name == "pto.wait_event" || name == "pto.tsync";
}

static bool isPtoInstrOp(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name.starts_with("pto.") && !isPtoMetaOp(op);
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

static std::string stripIndexing(std::string s) {
  auto l = s.find('[');
  if (l == std::string::npos)
    return s;
  return s.substr(0, l);
}

static bool isTileTypeString(llvm::StringRef typeStr) { return typeStr.starts_with("!pto.tile"); }

static llvm::StringRef parseTileLocFromType(llvm::StringRef typeStr) {
  // Very small parser: `!pto.tile<..., loc=Vec, ...>`.
  auto locPos = typeStr.find("loc=");
  if (locPos == llvm::StringRef::npos)
    return "";
  auto start = locPos + 4;
  auto end = typeStr.find_first_of(",>", start);
  if (end == llvm::StringRef::npos)
    end = typeStr.size();
  return typeStr.slice(start, end).trim();
}

static llvm::StringRef pipeForOpEnum(llvm::StringRef opEnum) {
  // Keep this minimal: only enums we may synthesize.
  if (opEnum == "TLOAD")
    return "MTE2";
  if (opEnum == "TSTORE_VEC" || opEnum == "TSTORE_MAT")
    return "MTE3";
  if (opEnum == "TSTORE_ACC")
    return "FIX";
  if (opEnum == "TMATMUL")
    return "M";
  if (opEnum == "TMOV_V2V")
    return "V";
  if (opEnum == "TMOV_V2M" || opEnum == "TEXTRACT_V2M" || opEnum == "TMOV_A2V" || opEnum == "TMOV_A2M")
    return "FIX";
  if (opEnum == "TMOV_M2B" || opEnum == "TMOV_M2L" || opEnum == "TMOV_M2R" || opEnum == "TEXTRACT_M2LR")
    return "MTE1";
  if (opEnum == "TMOV_M2S")
    return "FIX";
  if (opEnum == "TCI")
    return "S";
  // Default: most remaining PTO ops are vector pipe on A2/A3 (see `include/pto/npu/a2a3/TSync.hpp`).
  if (opEnum.starts_with("T"))
    return "V";
  return "";
}

static std::string uppercaseOpEnum(llvm::StringRef opcode) {
  std::string out;
  out.reserve(opcode.size());
  for (char ch : opcode)
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
  return out;
}

static std::string opcodeToOpEnum(llvm::StringRef opcode, mlir::Operation *op,
                                  const std::map<std::string, std::string> &argTypes) {
  // Minimal mapping for the prototype. Must return an enum value from `pto::Op`.
  if (opcode == "tload")
    return "TLOAD";

  if (opcode == "tstore") {
    auto operands = readOperands(op);
    if (operands.size() != 2)
      return "";
    auto src = stripIndexing(operands[1]);
    auto it = argTypes.find(src);
    if (it == argTypes.end())
      return "TSTORE_VEC";
    auto loc = parseTileLocFromType(it->second);
    if (loc == "Acc")
      return "TSTORE_ACC";
    if (loc == "Mat")
      return "TSTORE_MAT";
    return "TSTORE_VEC";
  }

  if (opcode == "tmatmul")
    return "TMATMUL";
  if (opcode == "tmatmul_acc")
    return "TMATMUL";

  if (opcode == "tmov") {
    auto operands = readOperands(op);
    if (operands.size() != 2)
      return "";
    auto dst = stripIndexing(operands[0]);
    auto src = stripIndexing(operands[1]);
    auto dstIt = argTypes.find(dst);
    auto srcIt = argTypes.find(src);
    if (dstIt == argTypes.end() || srcIt == argTypes.end())
      return "";
    auto dstLoc = parseTileLocFromType(dstIt->second);
    auto srcLoc = parseTileLocFromType(srcIt->second);

    if (srcLoc == "Vec" && dstLoc == "Vec")
      return "TMOV_V2V";
    if (srcLoc == "Vec" && dstLoc == "Mat")
      return "TMOV_V2M";
    if (srcLoc == "Mat" && dstLoc == "Bias")
      return "TMOV_M2B";
    if (srcLoc == "Mat" && dstLoc == "Left")
      return "TMOV_M2L";
    if (srcLoc == "Mat" && dstLoc == "Right")
      return "TMOV_M2R";
    if (srcLoc == "Mat" && (dstLoc == "ScaleLeft" || dstLoc == "ScaleRight" || dstLoc == "Scaling"))
      return "TMOV_M2S";
    if (srcLoc == "Acc" && dstLoc == "Vec")
      return "TMOV_A2V";
    if (srcLoc == "Acc" && dstLoc == "Mat")
      return "TMOV_A2M";
    return "";
  }

  // Default: PTO ISA op enums are typically just uppercase of the mnemonic.
  // This allows `--insert-events` to cover more vector ops (TMULS/TEXP/TROWMAX/...).
  if (opcode.starts_with("t"))
    return uppercaseOpEnum(opcode);

  return "";
}

struct DefInfo {
  mlir::Operation *op = nullptr;
  std::string opEnum;
  std::string pipe;
};

struct TileSyncState {
  DefInfo def;
  // Consumer pipes that have already waited for `def` in the current static scope.
  // This avoids generating multiple `wait_flag` for the same `(set_flag, token)` which
  // can deadlock on hardware that treats `wait_flag` as consuming a token.
  std::set<std::string> waitedPipes;
};

static bool opcodeDefinesTile(llvm::StringRef opcode) {
  // Heuristic for the prototype: most `t*` ops define their first operand (DPS).
  // Exceptions:
  // - tstore: writes to GM, does not define a tile
  // - tsync/record_event: meta
  // - tassign: binds address, not a data-producing tile op
  if (!opcode.starts_with("t"))
    return false;
  return opcode != "tstore" && opcode != "tsync" && opcode != "record_event" && opcode != "tassign";
}

static std::vector<std::string> opcodeTileUses(llvm::StringRef opcode, mlir::Operation *op,
                                               const std::map<std::string, std::string> &argTypes) {
  auto operands = readOperands(op);
  if (operands.empty())
    return {};

  // tstore: operands = [dstTensor, srcTile]
  if (opcode == "tstore") {
    if (operands.size() >= 2)
      return {stripIndexing(trim(operands[1]))};
    return {};
  }

  // tload: operands = [dstTile, srcTensor]
  if (opcode == "tload")
    return {};

  // Generic `t*` DPS-like ops: operands = [dst, src0, src1, ...]
  if (opcode.starts_with("t")) {
    std::vector<std::string> uses;
    for (size_t i = 1; i < operands.size(); ++i) {
      auto base = stripIndexing(trim(operands[i]));
      if (argTypes.count(base) && isTileTypeString(argTypes.at(base)))
        uses.push_back(base);
      else if (!base.empty() && base[0] == '%')
        uses.push_back(base); // best-effort: unknown symbols are treated as tiles
    }
    return uses;
  }

  return {};
}

static void insertEventOp(mlir::OpBuilder &b, mlir::Location loc, llvm::StringRef kind, llvm::StringRef srcOp,
                          llvm::StringRef dstOp, llvm::StringRef token) {
  mlir::OperationState st(loc, ("pto." + kind).str());
  st.addAttribute("src_op", b.getStringAttr(srcOp));
  st.addAttribute("dst_op", b.getStringAttr(dstOp));
  st.addAttribute("token", b.getStringAttr(token));
  b.create(st);
}

static bool hasEquivalentRecordEventAfter(mlir::Operation *producer, llvm::StringRef srcOp, llvm::StringRef dstOp,
                                          llvm::StringRef token) {
  for (auto *n = producer->getNextNode(); n && n->getName().getStringRef() == "pto.record_event";
       n = n->getNextNode()) {
    auto srcA = n->getAttrOfType<mlir::StringAttr>("src_op");
    auto dstA = n->getAttrOfType<mlir::StringAttr>("dst_op");
    auto tokA = n->getAttrOfType<mlir::StringAttr>("token");
    if (!srcA || !dstA || !tokA)
      continue;
    if (srcA.getValue() == srcOp && dstA.getValue() == dstOp && tokA.getValue() == token)
      return true;
  }
  return false;
}

static bool hasEquivalentWaitEventBefore(mlir::Operation *consumer, llvm::StringRef srcOp, llvm::StringRef dstOp,
                                         llvm::StringRef token) {
  for (auto *p = consumer->getPrevNode(); p && p->getName().getStringRef() == "pto.wait_event"; p = p->getPrevNode()) {
    auto srcA = p->getAttrOfType<mlir::StringAttr>("src_op");
    auto dstA = p->getAttrOfType<mlir::StringAttr>("dst_op");
    auto tokA = p->getAttrOfType<mlir::StringAttr>("token");
    if (!srcA || !dstA || !tokA)
      continue;
    if (srcA.getValue() == srcOp && dstA.getValue() == dstOp && tokA.getValue() == token)
      return true;
  }
  return false;
}

struct InsertEventsPass : public mlir::PassWrapper<InsertEventsPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!module.getBody())
      return;

    // Collect tile types so we can choose TMOV/TSTORE enum variants correctly.
    std::map<std::string, std::string> argTypes;
    module.walk([&](mlir::Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "pto.arg") {
        auto n = op->getAttrOfType<mlir::StringAttr>("name");
        auto t = op->getAttrOfType<mlir::StringAttr>("type");
        if (!n || !t)
          return;
        argTypes[n.getValue().str()] = t.getValue().str();
        return;
      }
      if (name == "pto.alloc_tile") {
        auto operands = readOperands(op);
        if (operands.empty())
          return;
        auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig");
        if (!typeSig)
          return;
        argTypes[trim(operands[0])] = typeSig.getValue().str();
        return;
      }
    });

    mlir::OpBuilder b(module.getContext());
    int loopId = 0;

    auto opOrNestedHasPipe = [&](mlir::Operation *op, llvm::StringRef wantPipe) -> bool {
      if (!op)
        return false;
      bool found = false;
      auto visit = [&](mlir::Operation *nested) {
        if (found)
          return;
        if (!isPtoInstrOp(nested))
          return;
        auto opcode = stripDialect(nested->getName().getStringRef());
        auto opEnum = opcodeToOpEnum(opcode, nested, argTypes);
        if (opEnum.empty())
          return;
        auto pipe = pipeForOpEnum(opEnum);
        if (pipe == wantPipe)
          found = true;
      };
      // Check op itself first.
      visit(op);
      if (found)
        return true;
      for (auto &r : op->getRegions()) {
        if (r.empty())
          continue;
        r.walk(visit);
        if (found)
          return true;
      }
      return found;
    };

    // Token allocator:
    // - Tokens are only 0..7 per (srcPipe,dstPipe) channel (see EventIdCounter<SrcPipe,DstPipe>).
    // - IMPORTANT: tokens are pipe-scoped, not opcode-scoped. Different PTO op enums may map to
    //   the same pipe pair (e.g. TMOV_M2L and TMOV_M2R are both MTE1), so key allocation must
    //   be based on pipes to avoid collisions that can deadlock.
    // - We keep a stable mapping per (srcPipe, dstPipe, key) to avoid reuse hazards within the same channel.
    // - If a channel needs >8 distinct keys, we fall back to a hashed slot (still bounded).
    std::map<std::tuple<std::string, std::string, std::string>, std::string> tokenByEdgeKey;
    std::map<std::pair<std::string, std::string>, int> nextTokenByEdge;
    auto allocTokenFor = [&](llvm::StringRef srcOp, llvm::StringRef dstOp, llvm::StringRef key) -> std::string {
      auto srcPipe = pipeForOpEnum(srcOp);
      auto dstPipe = pipeForOpEnum(dstOp);
      // If we cannot infer pipes (shouldn't happen for supported ops), fall back to opcode pair.
      std::string srcCh = !srcPipe.empty() ? srcPipe.str() : srcOp.str();
      std::string dstCh = !dstPipe.empty() ? dstPipe.str() : dstOp.str();

      auto k = std::make_tuple(srcCh, dstCh, key.str());
      auto it = tokenByEdgeKey.find(k);
      if (it != tokenByEdgeKey.end())
        return it->second;

      auto ch = std::make_pair(srcCh, dstCh);
      int next = nextTokenByEdge[ch];
      if (next < 8) {
        nextTokenByEdge[ch] = next + 1;
        return tokenByEdgeKey.emplace(k, std::to_string(next)).first->second;
      }

      // Fallback: keep within [0,7] but avoid crashing. This may reduce overlap.
      uint32_t h = 2166136261u;
      for (char c : key.str()) {
        h ^= static_cast<uint8_t>(c);
        h *= 16777619u;
      }
      return tokenByEdgeKey.emplace(k, std::to_string(h % 8)).first->second;
    };

    auto processBlock = [&](mlir::Block &block, std::map<std::string, TileSyncState> &tileState, auto &&self) -> void {
      for (auto it = block.begin(); it != block.end(); ++it) {
        auto *consumer = &*it;
        auto opname = consumer->getName().getStringRef();
        if (opname == "scf.for") {
          int thisLoopId = loopId++;
          // Loop-carried reuse hazards:
          // - Many real kernels reuse the same tile storage across loop iterations.
          // - Since `set_flag` is pipe-scoped (not object-scoped), we insert a conservative
          //   per-iteration handshake to prevent overwriting tiles that may still be in-flight.
          //
          // This implements the spec's "Reverse dependency must be explicit" rule (WAR/WAW hazards)
          // in a conservative way for common pipelines:
          //   - M -> MTE1 (protect Left/Right reuse across TMATMUL)
          //   - MTE1 -> MTE2 (protect Mat/L1 reuse across TMOV)
          //
          // Pattern: prime in preheader, then {wait at loop start; record at loop end} each iter.
          bool hasM = false;
          bool hasMte1 = false;
          bool hasMte2 = false;
          for (auto &r : consumer->getRegions()) {
            if (r.empty())
              continue;
            r.walk([&](mlir::Operation *op) {
              if (!isPtoInstrOp(op))
                return;
              auto opcode = stripDialect(op->getName().getStringRef());
              auto opEnum = opcodeToOpEnum(opcode, op, argTypes);
              if (opEnum.empty())
                return;
              auto pipe = pipeForOpEnum(opEnum);
              if (pipe == "M")
                hasM = true;
              else if (pipe == "MTE1")
                hasMte1 = true;
              else if (pipe == "MTE2")
                hasMte2 = true;
            });
          }

          // Insert priming events in the preheader (before the scf.for op).
          b.setInsertionPoint(consumer);
          if (hasM && hasMte1) {
            auto key = ("loop_m_to_mte1#" + std::to_string(thisLoopId));
            auto tok = allocTokenFor("TMATMUL", "TMOV_M2L", key);
            insertEventOp(b, consumer->getLoc(), "record_event", "TMATMUL", "TMOV_M2L", tok);
          }
          if (hasMte1 && hasMte2) {
            auto key = ("loop_mte1_to_mte2#" + std::to_string(thisLoopId));
            auto tok = allocTokenFor("TMOV_M2L", "TLOAD", key);
            insertEventOp(b, consumer->getLoc(), "record_event", "TMOV_M2L", "TLOAD", tok);
          }
          if (hasMte1 && hasM) {
            // Loop-carried RAW: a common ping-pong GEMM pattern prefetches (MTE1) data for the *next*
            // iteration and consumes it (M) in the following iteration. Insert a per-iteration MTE1->M
            // handshake so TMATMUL does not read tiles that are still being produced by TMOV.
            //
            // NOTE: Use a loop-unique key so nested loops don't accidentally share the same token.
            auto key = ("loop_mte1_to_m#" + std::to_string(thisLoopId));
            auto tok = allocTokenFor("TMOV_M2L", "TMATMUL", key);
            insertEventOp(b, consumer->getLoc(), "record_event", "TMOV_M2L", "TMATMUL", tok);
          }

          // Insert per-iteration waits/records inside the loop body (front block).
          for (auto &r : consumer->getRegions()) {
            if (r.empty())
              continue;
            auto &body = r.front();
            auto *term = body.getTerminator();

            if (hasM && hasMte1) {
              auto key = ("loop_m_to_mte1#" + std::to_string(thisLoopId));
              auto tok = allocTokenFor("TMATMUL", "TMOV_M2L", key);
              b.setInsertionPointToStart(&body);
              insertEventOp(b, consumer->getLoc(), "wait_event", "TMATMUL", "TMOV_M2L", tok);

              // Record after the last op in the loop body that (transitively) executes TMATMUL (PIPE_M).
              // This avoids placing `set_flag(PIPE_M, PIPE_MTE1, ...)` before the loop's matmul, which
              // would be one-iteration behind and can cause L0A/L0B read/write conflicts when ping-pong
              // buffers are reused across iterations.
              mlir::Operation *lastMCarrier = nullptr;
              for (auto &opInBody : body) {
                mlir::Operation *cur = &opInBody;
                if (cur == term)
                  break;
                if (opOrNestedHasPipe(cur, "M"))
                  lastMCarrier = cur;
              }
              if (lastMCarrier) {
                b.setInsertionPointAfter(lastMCarrier);
                insertEventOp(b, consumer->getLoc(), "record_event", "TMATMUL", "TMOV_M2L", tok);
              } else {
                // Fallback: still insert before the terminator to keep token balance.
                b.setInsertionPoint(term);
                insertEventOp(b, consumer->getLoc(), "record_event", "TMATMUL", "TMOV_M2L", tok);
              }
            }
            if (hasMte1 && hasMte2) {
              auto key = ("loop_mte1_to_mte2#" + std::to_string(thisLoopId));
              auto tok = allocTokenFor("TMOV_M2L", "TLOAD", key);
              b.setInsertionPointToStart(&body);
              insertEventOp(b, consumer->getLoc(), "wait_event", "TMOV_M2L", "TLOAD", tok);
              b.setInsertionPoint(term);
              insertEventOp(b, consumer->getLoc(), "record_event", "TMOV_M2L", "TLOAD", tok);
            }
            if (hasMte1 && hasM) {
              auto key = ("loop_mte1_to_m#" + std::to_string(thisLoopId));
              auto tok = allocTokenFor("TMOV_M2L", "TMATMUL", key);
              b.setInsertionPointToStart(&body);
              insertEventOp(b, consumer->getLoc(), "wait_event", "TMOV_M2L", "TMATMUL", tok);
              b.setInsertionPoint(term);
              insertEventOp(b, consumer->getLoc(), "record_event", "TMOV_M2L", "TMATMUL", tok);
            }
          }

          // Process the loop region with a snapshot of the current tile state, then conservatively
          // merge by invalidating tile defs (loop may execute 0 times).
          for (auto &r : consumer->getRegions())
            if (!r.empty()) {
              auto inner = tileState;
              self(r.front(), inner, self);
            }
          continue;
        }
        if (opname == "scf.if") {
          // Process then/else with independent states; merge conservatively.
          std::map<std::string, TileSyncState> thenState = tileState;
          std::map<std::string, TileSyncState> elseState = tileState;
          if (consumer->getNumRegions() >= 1 && !consumer->getRegion(0).empty())
            self(consumer->getRegion(0).front(), thenState, self);
          if (consumer->getNumRegions() >= 2 && !consumer->getRegion(1).empty())
            self(consumer->getRegion(1).front(), elseState, self);

          std::set<std::string> keys;
          for (auto &kv : thenState)
            keys.insert(kv.first);
          for (auto &kv : elseState)
            keys.insert(kv.first);

          std::map<std::string, TileSyncState> merged = tileState;
          for (auto &k : keys) {
            auto itT = thenState.find(k);
            auto itE = elseState.find(k);
            if (itT == thenState.end() || itE == elseState.end()) {
              // Defined/updated in only one branch: invalidate to avoid generating unmatched waits.
              merged.erase(k);
              continue;
            }
            auto &t = itT->second;
            auto &e = itE->second;
            if (t.def.opEnum == e.def.opEnum && t.def.pipe == e.def.pipe && !t.def.opEnum.empty() && !t.def.pipe.empty()) {
              TileSyncState out;
              // Anchor to the scf.if op itself so the record_event is unconditional.
              out.def = DefInfo{consumer, t.def.opEnum, t.def.pipe};
              out.waitedPipes.clear();
              out.waitedPipes.insert(out.def.pipe);
              merged[k] = std::move(out);
              continue;
            }
            merged.erase(k);
          }
          tileState = std::move(merged);
          continue;
        }

        if (!isPtoInstrOp(consumer))
          continue;

        auto consOpcode = stripDialect(consumer->getName().getStringRef());
        auto consEnum = opcodeToOpEnum(consOpcode, consumer, argTypes);
        if (consEnum.empty())
          continue;
        auto consPipe = pipeForOpEnum(consEnum);
        if (consPipe.empty())
          continue;

        // Insert waits for cross-pipe RAW tile dependencies (producer -> consumer).
        // This is conservative and pipe-scoped per the hardware contract: `set_flag` releases all
        // prior ops on srcPipe, so we must avoid inserting multiple `wait_flag` on the same token.
        for (auto &useSym : opcodeTileUses(consOpcode, consumer, argTypes)) {
          auto use = stripIndexing(trim(useSym));
          auto defIt = tileState.find(use);
          if (defIt == tileState.end())
            continue;
          auto &defState = defIt->second;
          auto &def = defState.def;
          if (!def.op || def.pipe.empty() || def.opEnum.empty())
            continue;
          if (def.pipe == consPipe)
            continue;

          // Only wait once per (tile, consumer pipe) for the current definition site.
          if (defState.waitedPipes.count(consPipe.str()))
            continue;
          auto token = allocTokenFor(def.opEnum, consEnum, use);

          // Ensure record_event exists on the producer path.
          if (!hasEquivalentRecordEventAfter(def.op, def.opEnum, consEnum, token)) {
            b.setInsertionPointAfter(def.op);
            insertEventOp(b, def.op->getLoc(), "record_event", def.opEnum, consEnum, token);
          }

          // Ensure wait_event exists before consumer.
          if (!hasEquivalentWaitEventBefore(consumer, def.opEnum, consEnum, token)) {
            b.setInsertionPoint(consumer);
            insertEventOp(b, consumer->getLoc(), "wait_event", def.opEnum, consEnum, token);
          }
          defState.waitedPipes.insert(consPipe.str());
        }

        // Update last-def for tile results.
        if (opcodeDefinesTile(consOpcode)) {
          auto operands = readOperands(consumer);
          if (!operands.empty()) {
            auto dst = stripIndexing(trim(operands[0]));
            TileSyncState st;
            st.def = DefInfo{consumer, consEnum, consPipe.str()};
            st.waitedPipes.insert(consPipe.str());
            tileState[dst] = std::move(st);
          }
        }
      }
    };

    std::map<std::string, TileSyncState> tileState;
    processBlock(*module.getBody(), tileState, processBlock);
  }

  llvm::StringRef getArgument() const final { return "ptoas-insert-events"; }
  llvm::StringRef getDescription() const final {
    return "Insert pto.record_event + pto.wait_event for cross-pipe tile dependencies (prototype).";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertEventsPass() {
  return std::make_unique<InsertEventsPass>();
}

} // namespace ptoas
