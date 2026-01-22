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
  if (opEnum == "TADD" || opEnum == "TMUL" || opEnum == "TSUB" || opEnum == "TCVT" || opEnum == "TMRGSORT" ||
      opEnum == "TSORT32")
    return "V";
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
  return "";
}

static llvm::StringRef opcodeToOpEnum(llvm::StringRef opcode, mlir::Operation *op,
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

  if (opcode == "tadd")
    return "TADD";
  if (opcode == "tmul")
    return "TMUL";
  if (opcode == "tsub")
    return "TSUB";
  if (opcode == "tcvt")
    return "TCVT";
  if (opcode == "tmrgsort")
    return "TMRGSORT";
  if (opcode == "tsort32")
    return "TSORT32";
  if (opcode == "tmatmul")
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

  return "";
}

struct DefInfo {
  mlir::Operation *op = nullptr;
  std::string opEnum;
  std::string pipe;
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
    int nextToken = 0;
    auto allocToken = [&]() -> std::string {
      // Hardware typically caps event id count; keep tokens in [0,7] for the prototype.
      int tok = nextToken++ % 8;
      return std::to_string(tok);
    };

    // Dedup tokens per semantic edge from a specific producer: (producer op, src_op, dst_op) -> token.
    std::map<std::tuple<mlir::Operation *, std::string, std::string>, std::string> edgeToken;

    auto processBlock = [&](mlir::Block &block, auto &&self) -> void {
      std::map<std::string, DefInfo> lastDef;

      for (auto it = block.begin(); it != block.end(); ++it) {
        auto *consumer = &*it;
        if (consumer->getName().getStringRef() == "scf.for" || consumer->getName().getStringRef() == "scf.if") {
          for (auto &r : consumer->getRegions())
            if (!r.empty())
              self(r.front(), self);
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

        // Insert waits for cross-pipe tile dependencies.
        for (auto &useSym : opcodeTileUses(consOpcode, consumer, argTypes)) {
          auto use = stripIndexing(trim(useSym));
          auto defIt = lastDef.find(use);
          if (defIt == lastDef.end())
            continue;
          auto &def = defIt->second;
          if (!def.op || def.pipe.empty() || def.opEnum.empty())
            continue;
          if (def.pipe == consPipe)
            continue;

          // Dedup at the semantic edge level (SrcOp,DstOp) and keep token count bounded.
          // Key: (producer op ptr, src_op, dst_op).
          auto key = std::make_tuple(def.op, def.opEnum, consEnum.str());
          auto itTok = edgeToken.find(key);
          if (itTok == edgeToken.end())
            itTok = edgeToken.emplace(key, allocToken()).first;
          auto token = itTok->second;

          // Ensure record_event exists on the producer path.
          if (!hasEquivalentRecordEventAfter(def.op, def.opEnum, consEnum, token)) {
            b.setInsertionPointAfter(def.op);
            mlir::OperationState st(def.op->getLoc(), "pto.record_event");
            st.addAttribute("src_op", b.getStringAttr(def.opEnum));
            st.addAttribute("dst_op", b.getStringAttr(consEnum));
            st.addAttribute("token", b.getStringAttr(token));
            b.create(st);
          }

          // Ensure wait_event exists before consumer.
          if (!hasEquivalentWaitEventBefore(consumer, def.opEnum, consEnum, token)) {
            b.setInsertionPoint(consumer);
            mlir::OperationState st(consumer->getLoc(), "pto.wait_event");
            st.addAttribute("src_op", b.getStringAttr(def.opEnum));
            st.addAttribute("dst_op", b.getStringAttr(consEnum));
            st.addAttribute("token", b.getStringAttr(token));
            b.create(st);
          }
        }

        // Update last-def for tile results.
        if (opcodeDefinesTile(consOpcode)) {
          auto operands = readOperands(consumer);
          if (!operands.empty()) {
            auto dst = stripIndexing(trim(operands[0]));
            lastDef[dst] = DefInfo{consumer, consEnum.str(), consPipe.str()};
          }
        }
      }
    };

    processBlock(*module.getBody(), processBlock);
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
