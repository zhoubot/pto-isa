#include "ptoas/Passes.h"
#include "ptoas/ProtoAttrs.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <map>
#include <optional>
#include <sstream>
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

static std::string normalizeHexAddr(const std::string &s) {
  auto v = parseIntLiteralOrZero(s);
  std::ostringstream ss;
  ss << "0x" << std::hex << v;
  return ss.str();
}

static llvm::StringRef stripDialect(llvm::StringRef opName) {
  if (auto dot = opName.find('.'); dot != llvm::StringRef::npos)
    return opName.drop_front(dot + 1);
  return opName;
}

static bool isPtoMetaOp(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "pto.arg" || name == "pto.const" || name == "pto.make_tensor_view" || name == "pto.subview" ||
         name == "pto.partition_view" || name == "pto.alloc_tile" || name == "pto.record_event" ||
         name == "pto.wait_event" || name == "pto.tsync";
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
    // Acc -> GM store uses FIX pipe (see A5 ST `tstore_acc2gm` patterns).
    return "FIX";
  if (opEnum == "TRESHAPE")
    return "S";
  if (opEnum == "TMATMUL")
    return "M";
  if (opEnum == "TMATMUL_MX")
    return "M";
  if (opEnum == "TMATMUL_BIAS")
    return "M";
  if (opEnum == "TMOV_V2V")
    return "V";
  if (opEnum == "TMOV_V2M" || opEnum == "TEXTRACT_V2M" || opEnum == "TMOV_A2V" || opEnum == "TMOV_A2M")
    return "FIX";
  if (opEnum == "TEXTRACT_A2M" || opEnum == "TINSERT_A2M")
    return "FIX";
  // Fallback spelling used when we can't disambiguate TEXTRACT/TINSERT variants from operand metadata.
  // These operations are implemented on PIPE_FIX for Acc->Mat and Vec->Mat paths on A2/A3.
  if (opEnum == "TEXTRACT" || opEnum == "TINSERT")
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
  if (opcode == "tprefetch")
    return "TLOAD";
  if (opcode == "mgather")
    return "TLOAD";
  if (opcode == "mscatter")
    return "TSTORE_VEC";

  if (opcode == "tpop")
    return "TLOAD";

  if (opcode == "tstore") {
    auto operands = readOperands(op);
    // Support indexed form: [dstTensor, (optional indices...), srcTile].
    if (operands.size() < 2)
      return "";
    auto src = stripIndexing(operands.back());
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

  if (opcode == "tpush") {
    auto operands = readOperands(op);
    // Prototype: operands = [dstTensor, srcTile, token]
    if (operands.size() < 2)
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
  if (opcode == "tmatmul_mx")
    return "TMATMUL_MX";

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

  if (opcode == "textract") {
    auto operands = readOperands(op);
    // Prototype: operands = [dstTile, srcTile, (optional indexRow), (optional indexCol)].
    if (operands.size() < 2)
      return "";
    auto dst = stripIndexing(operands[0]);
    auto src = stripIndexing(operands[1]);
    auto dstIt = argTypes.find(dst);
    auto srcIt = argTypes.find(src);
    if (dstIt == argTypes.end() || srcIt == argTypes.end())
      return "";
    auto dstLoc = parseTileLocFromType(dstIt->second);
    auto srcLoc = parseTileLocFromType(srcIt->second);
    if (srcLoc == "Vec" && dstLoc == "Mat")
      return "TEXTRACT_V2M";
    if (srcLoc == "Mat" && (dstLoc == "Left" || dstLoc == "Right"))
      return "TEXTRACT_M2LR";
    if (srcLoc == "Acc" && dstLoc == "Mat")
      return "TEXTRACT_A2M";
    return "";
  }

  if (opcode == "tinsert") {
    auto operands = readOperands(op);
    // Prototype: operands = [dstTile, srcTile, (optional indexRow), (optional indexCol)].
    if (operands.size() < 2)
      return "";
    auto dst = stripIndexing(operands[0]);
    auto src = stripIndexing(operands[1]);
    auto dstIt = argTypes.find(dst);
    auto srcIt = argTypes.find(src);
    if (dstIt == argTypes.end() || srcIt == argTypes.end())
      return "";
    auto dstLoc = parseTileLocFromType(dstIt->second);
    auto srcLoc = parseTileLocFromType(srcIt->second);
    if (srcLoc == "Acc" && dstLoc == "Mat")
      return "TINSERT_A2M";
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
  if (opcode == "mgather")
    return true;
  if (!opcode.starts_with("t"))
    return false;
  return opcode != "tstore" && opcode != "tpush" && opcode != "tsync" && opcode != "record_event" && opcode != "tassign";
}

static std::vector<std::string> opcodeTileUses(llvm::StringRef opcode, mlir::Operation *op,
                                               const std::map<std::string, std::string> &argTypes) {
  auto operands = readOperands(op);
  if (operands.empty())
    return {};

  // tpush: operands = [dstTensor, srcTile, token]
  if (opcode == "tpush") {
    if (operands.size() >= 2)
      return {stripIndexing(trim(operands[1]))};
    return {};
  }

  // tstore: operands = [dstTensor, (optional indices...), srcTile]
  if (opcode == "tstore") {
    if (operands.size() >= 2)
      return {stripIndexing(trim(operands.back()))};
    return {};
  }

  // tload: operands = [dstTile, srcTensor]
  if (opcode == "tload" || opcode == "tpop")
    return {};

  // Generic `t*` / mgather/mscatter DPS-like ops: operands = [dst, src0, src1, ...]
  if (opcode.starts_with("t") || opcode == "mgather" || opcode == "mscatter") {
    std::vector<std::string> uses;
    for (size_t i = 1; i < operands.size(); ++i) {
      auto base = stripIndexing(trim(operands[i]));
      if (argTypes.count(base) && isTileTypeString(argTypes.at(base)))
        uses.push_back(base);
    }
    return uses;
  }

  return {};
}

static mlir::Operation *insertEventOp(mlir::OpBuilder &b, mlir::Operation *anchor, mlir::Location loc,
                                      llvm::StringRef kind,
                                      llvm::StringRef srcOp, llvm::StringRef dstOp, llvm::StringRef token) {
  mlir::OperationState st(loc, ("pto." + kind).str());
  st.addAttribute("src_op", b.getStringAttr(srcOp));
  st.addAttribute("dst_op", b.getStringAttr(dstOp));
  st.addAttribute("token", b.getStringAttr(token));
  if (anchor) {
    if (auto stage = anchor->getAttrOfType<mlir::StringAttr>(kStageAttr))
      st.addAttribute(kStageAttr, stage);
  }
  return b.create(st);
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

static bool hasFencePairAfter(mlir::Operation *producer, llvm::StringRef srcOp, llvm::StringRef dstOp) {
  auto *rec = producer ? producer->getNextNode() : nullptr;
  if (!rec || rec->getName().getStringRef() != "pto.record_event")
    return false;
  auto recSrcA = rec->getAttrOfType<mlir::StringAttr>("src_op");
  auto recDstA = rec->getAttrOfType<mlir::StringAttr>("dst_op");
  auto recTokA = rec->getAttrOfType<mlir::StringAttr>("token");
  if (!recSrcA || !recDstA || !recTokA)
    return false;
  if (recSrcA.getValue() != srcOp || recDstA.getValue() != dstOp)
    return false;

  auto *wait = rec->getNextNode();
  if (!wait || wait->getName().getStringRef() != "pto.wait_event")
    return false;
  auto waitSrcA = wait->getAttrOfType<mlir::StringAttr>("src_op");
  auto waitDstA = wait->getAttrOfType<mlir::StringAttr>("dst_op");
  auto waitTokA = wait->getAttrOfType<mlir::StringAttr>("token");
  if (!waitSrcA || !waitDstA || !waitTokA)
    return false;
  if (waitSrcA.getValue() != srcOp || waitDstA.getValue() != dstOp)
    return false;
  return waitTokA.getValue() == recTokA.getValue();
}

static bool hasFenceBefore(mlir::Operation *consumer, llvm::StringRef srcOp, llvm::StringRef dstOp) {
  // Look backward through a contiguous "meta" region (record/wait/tsync) immediately before `consumer`.
  // If any wait_event matches the requested (src_op, dst_op), treat the fence as already present.
  for (auto *p = consumer ? consumer->getPrevNode() : nullptr; p; p = p->getPrevNode()) {
    auto name = p->getName().getStringRef();
    if (name == "pto.tsync" || name == "pto.record_event")
      continue;
    if (name == "pto.wait_event") {
      auto s = p->getAttrOfType<mlir::StringAttr>("src_op");
      auto d = p->getAttrOfType<mlir::StringAttr>("dst_op");
      if (s && d && s.getValue() == srcOp && d.getValue() == dstOp)
        return true;
      continue;
    }
    break;
  }
  return false;
}

struct InsertEventsPass : public mlir::PassWrapper<InsertEventsPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    if (!module.getBody())
      return;

    // Collect tile types so we can choose TMOV/TSTORE enum variants correctly.
    // Also collect constants so we can prove some loops execute at least once.
    std::map<std::string, std::string> argTypes;
    std::map<std::string, std::string> tileKeys;
    std::map<std::string, std::string> constMap;
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
        auto tileName = trim(operands[0]);
        auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig");
        if (!typeSig)
          return;
        auto typeStr = typeSig.getValue().str();
        argTypes[tileName] = typeStr;

        // Track alias keys for tiles that share the same underlying buffer address.
        // Key format: "<loc>@<addr>" (e.g. "Right@0x8000"). This lets the sync insertion pass
        // reason about hazards on reused physical buffers, even when multiple tile variables
        // are assigned to the same address slot by the address allocator.
        std::string addrStr;
        if (operands.size() >= 2)
          addrStr = trim(operands[1]);
        auto loc = parseTileLocFromType(typeStr);
        if (!addrStr.empty() && !loc.empty()) {
          tileKeys[tileName] = (loc.str() + "@" + normalizeHexAddr(addrStr));
        } else {
          tileKeys[tileName] = tileName;
        }
        return;
      }
      if (name == "pto.const") {
        auto n = op->getAttrOfType<mlir::StringAttr>("name");
        auto v = op->getAttrOfType<mlir::StringAttr>("value");
        if (!n || !v)
          return;
        constMap[n.getValue().str()] = trim(v.getValue().str());
        return;
      }
    });

    // Lightweight constant propagation for index arithmetic so we can:
    // - prove `scf.for` loops execute (or execute >1 times)
    // - keep producer state across loops to insert correct fences for tstore/tmov
    //
    // We only fold a minimal subset of `pto.i*` ops used in loop-bound computations.
    auto resolveIntConst = [&](std::string s) -> std::optional<int64_t> {
      s = trim(s);
      if (s.empty())
        return std::nullopt;
      if (s[0] == '%') {
        auto it = constMap.find(s);
        if (it == constMap.end())
          return std::nullopt;
        s = trim(it->second);
      }
      char *end = nullptr;
      long long v = std::strtoll(s.c_str(), &end, 10);
      if (!end || *end != '\0')
        return std::nullopt;
      return static_cast<int64_t>(v);
    };

    auto tryFoldIndexOp = [&](mlir::Operation *op) -> void {
      auto name = stripDialect(op->getName().getStringRef());
      if (!(name == "iadd" || name == "isub" || name == "imul" || name == "idiv" || name == "irem"))
        return;
      auto operands = readOperands(op);
      if (operands.size() != 3)
        return;
      auto dst = trim(operands[0]);
      auto lhs = resolveIntConst(operands[1]);
      auto rhs = resolveIntConst(operands[2]);
      if (!lhs || !rhs)
        return;
      int64_t out = 0;
      if (name == "iadd") {
        out = *lhs + *rhs;
      } else if (name == "isub") {
        out = *lhs - *rhs;
      } else if (name == "imul") {
        out = (*lhs) * (*rhs);
      } else if (name == "idiv") {
        if (*rhs == 0)
          return;
        out = *lhs / *rhs;
      } else if (name == "irem") {
        if (*rhs == 0)
          return;
        out = *lhs % *rhs;
      } else {
        return;
      }
      constMap[dst] = std::to_string(out);
    };

    module.walk([&](mlir::Operation *op) { tryFoldIndexOp(op); });

    mlir::OpBuilder b(module.getContext());

    auto keyForTile = [&](const std::string &sym) -> std::string {
      auto it = tileKeys.find(sym);
      if (it != tileKeys.end())
        return it->second;
      return sym;
    };

    auto canonicalOpEnumForPipe = [&](llvm::StringRef pipe) -> llvm::StringRef {
      // Pick a stable representative op enum for each pipe so event insertion can
      // conservatively synchronize by pipe when the exact producer op is ambiguous
      // across control-flow merges.
      if (pipe == "MTE2")
        return "TLOAD";
      if (pipe == "MTE1")
        return "TMOV_M2L";
      if (pipe == "MTE3")
        return "TSTORE_VEC";
      if (pipe == "M")
        return "TMATMUL";
      if (pipe == "V")
        return "TMOV_V2V";
      if (pipe == "FIX")
        return "TSTORE_ACC";
      if (pipe == "S")
        return "TCI";
      return "";
    };

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
    // - Hardware only provides 8 event IDs per (srcPipe, dstPipe) channel.
    // - Always recycle tokens in [0..7] in a round-robin fashion.
    // - Insert record+wait as an adjacent pair so tokens are not live across dynamic paths.
    std::map<std::pair<std::string, std::string>, int> nextTokenByPipePair;
    auto allocTokenForPipes = [&](llvm::StringRef srcPipe, llvm::StringRef dstPipe) -> std::string {
      auto k = std::make_pair(srcPipe.str(), dstPipe.str());
      int &next = nextTokenByPipePair[k];
      int tok = next & 7;
      next = (next + 1) & 7;
      return std::to_string(tok);
    };

    auto insertFencePairBefore = [&](mlir::Operation *anchor, llvm::StringRef srcPipe, llvm::StringRef dstPipe,
                                     llvm::StringRef srcOpEnum, llvm::StringRef dstOpEnum) -> void {
      if (!anchor)
        return;
      if (srcPipe.empty() || dstPipe.empty())
        return;
      if (srcPipe == dstPipe)
        return;

      (void)srcOpEnum;
      (void)dstOpEnum;
      // Synchronization is pipe-based on A2/A3; use canonical op enums for stability.
      auto srcOp = canonicalOpEnumForPipe(srcPipe);
      auto dstOp = canonicalOpEnumForPipe(dstPipe);
      if (srcOp.empty() || dstOp.empty())
        return;

      // Keep the pass stable if re-run.
      if (hasFenceBefore(anchor, srcOp, dstOp))
        return;

      auto tok = allocTokenForPipes(srcPipe, dstPipe);
      b.setInsertionPoint(anchor);
      insertEventOp(b, anchor, anchor->getLoc(), "record_event", srcOp, dstOp, tok);
      insertEventOp(b, anchor, anchor->getLoc(), "wait_event", srcOp, dstOp, tok);
    };

    auto insertFencePairAfter = [&](mlir::Operation *anchor, llvm::StringRef srcPipe, llvm::StringRef dstPipe,
                                    llvm::StringRef srcOpEnum, llvm::StringRef dstOpEnum) -> void {
      if (!anchor)
        return;
      if (srcPipe.empty() || dstPipe.empty())
        return;
      if (srcPipe == dstPipe)
        return;

      (void)srcOpEnum;
      (void)dstOpEnum;
      // Synchronization is pipe-based on A2/A3; use canonical op enums for stability.
      auto srcOp = canonicalOpEnumForPipe(srcPipe);
      auto dstOp = canonicalOpEnumForPipe(dstPipe);
      if (srcOp.empty() || dstOp.empty())
        return;

      // Keep the pass stable if re-run.
      if (hasFencePairAfter(anchor, srcOp, dstOp))
        return;

      auto tok = allocTokenForPipes(srcPipe, dstPipe);
      b.setInsertionPointAfter(anchor);
      insertEventOp(b, anchor, anchor->getLoc(), "record_event", srcOp, dstOp, tok);
      insertEventOp(b, anchor, anchor->getLoc(), "wait_event", srcOp, dstOp, tok);
    };

    auto insertTsyncVBefore = [&](mlir::Operation *anchor) -> void {
      if (!anchor)
        return;
      // Keep the pass stable if re-run.
      if (auto *p = anchor->getPrevNode(); p && p->getName().getStringRef() == "pto.tsync") {
        auto pipeA = p->getAttrOfType<mlir::StringAttr>("pipe");
        auto pipe = pipeA ? trim(pipeA.getValue().str()) : std::string("V");
        if (pipe.empty() || pipe == "V")
          return;
      }
      b.setInsertionPoint(anchor);
      mlir::OperationState st(anchor->getLoc(), "pto.tsync");
      st.addAttribute("pipe", b.getStringAttr("V"));
      if (auto stage = anchor->getAttrOfType<mlir::StringAttr>(kStageAttr))
        st.addAttribute(kStageAttr, stage);
      b.create(st);
    };

	    auto processBlock = [&](mlir::Block &block, std::map<std::string, TileSyncState> &tileState, auto &&self) -> void {
        std::string lastPipe;
	      for (auto it = block.begin(); it != block.end(); ++it) {
	        auto *consumer = &*it;
		        auto opname = consumer->getName().getStringRef();
		        if (opname == "scf.for") {
	            lastPipe.clear();

		          // Process the loop region with a snapshot of the current tile state.
		          // If we can prove the loop executes at least once (constant bounds), propagate the
		          // resulting tile defs to the outer scope so consumers after the loop (e.g. tstore)
	          // can see producers inside the loop.
	          bool mustRunAtLeastOnce = false;
	          bool mayRunMoreThanOnce = true;
	          auto loopOperands = readOperands(consumer);
	          if (loopOperands.size() == 4) {
	            auto lb = resolveIntConst(loopOperands[1]);
	            auto ub = resolveIntConst(loopOperands[2]);
	            auto step = resolveIntConst(loopOperands[3]);
	            if (lb && ub && step && *step > 0) {
	              mustRunAtLeastOnce = (*lb < *ub);
	              // >=2 iterations if lb + step < ub
	              mayRunMoreThanOnce = ((*lb + *step) < *ub);
	            }
	          }

	          for (auto &r : consumer->getRegions())
	            if (!r.empty()) {
	              auto incoming = tileState;
	              auto inner = tileState;
	              self(r.front(), inner, self);
	              // Loop-carried hazards: the loop body executes repeatedly, so end-of-iter tile states
	              // feed into the next iteration's overwrites/uses. Re-run once with the end state as
	              // the incoming state to insert the missing cross-iteration fences (conservative).
	              if (mayRunMoreThanOnce) {
	                auto carried = inner;
	                self(r.front(), carried, self);
	                inner = std::move(carried);
	              }
	              if (mustRunAtLeastOnce) {
	                std::map<std::string, TileSyncState> out = std::move(inner);
	                for (auto &kv : out) {
	                  auto itIn = incoming.find(kv.first);
	                  bool changed = (itIn == incoming.end());
	                  if (!changed) {
	                    auto &a = itIn->second.def;
	                    auto &b2 = kv.second.def;
	                    changed = (a.op != b2.op) || (a.opEnum != b2.opEnum) || (a.pipe != b2.pipe);
	                  }
	                  if (!changed)
	                    continue;
	                  if (kv.second.def.pipe.empty() || kv.second.def.opEnum.empty())
	                    continue;
	                  auto canon = canonicalOpEnumForPipe(kv.second.def.pipe);
	                  if (!canon.empty())
	                    kv.second.def.opEnum = canon.str();
	                  kv.second.def.op = consumer; // anchor to the loop op (runs to completion)
	                  kv.second.waitedPipes.clear();
	                  kv.second.waitedPipes.insert(kv.second.def.pipe);
	                }
	                tileState = std::move(out);
	              }
	            }
	          continue;
	        }
        if (opname == "scf.if") {
          lastPipe.clear();
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
            auto itIn = tileState.find(k);

            // In this prototype, `%x = pto.tmov ...` style "rebindings" are treated as a mutable symbol table
            // rather than strict SSA. That means a tile may be *updated* in only one branch, while the other
            // branch keeps the previous value. Dropping the key here loses dependency info and can produce
            // missing waits in common ping-pong GEMM patterns (leading to L0 conflicts / deadlocks).

            auto considerMerge = [&](const TileSyncState &a, const TileSyncState &b) -> bool {
              if (a.def.pipe.empty() || b.def.pipe.empty())
                return false;
              if (a.def.pipe != b.def.pipe)
                return false;
              auto pipe = a.def.pipe;
              std::string opEnum;
              if (!a.def.opEnum.empty() && a.def.opEnum == b.def.opEnum) {
                opEnum = a.def.opEnum;
              } else if (!a.def.opEnum.empty() && pipeForOpEnum(a.def.opEnum) == pipe) {
                opEnum = a.def.opEnum;
              } else {
                auto canon = canonicalOpEnumForPipe(pipe);
                if (canon.empty())
                  return false;
                opEnum = canon.str();
              }

              TileSyncState out;
              // Anchor to the scf.if op itself so inserted record_event is unconditional after the merge.
              out.def = DefInfo{consumer, opEnum, pipe};
              out.waitedPipes.clear();
              out.waitedPipes.insert(pipe);
              merged[k] = std::move(out);
              return true;
            };

            if (itT == thenState.end() || itE == elseState.end()) {
              // Defined/updated in only one branch:
              // - If the symbol existed before the `scf.if`, treat the missing branch as "kept previous value"
              //   and merge by pipe.
              // - Otherwise, keep the known branch state conservatively. Dropping it loses dependency info and
              //   can produce missing waits in common ping-pong GEMM patterns (leading to L0 conflicts).
              if (itIn == tileState.end()) {
                const TileSyncState *only = (itT != thenState.end()) ? &itT->second : &itE->second;
                if (!only || only->def.pipe.empty()) {
                  merged.erase(k);
                  continue;
                }
                auto canon = canonicalOpEnumForPipe(only->def.pipe);
                if (canon.empty()) {
                  merged.erase(k);
                  continue;
                }
                TileSyncState out;
                out.def = DefInfo{consumer, canon.str(), only->def.pipe};
                out.waitedPipes.clear();
                out.waitedPipes.insert(only->def.pipe);
                merged[k] = std::move(out);
                continue;
              }
              const TileSyncState &t = (itT != thenState.end()) ? itT->second : itIn->second;
              const TileSyncState &e = (itE != elseState.end()) ? itE->second : itIn->second;
              if (!considerMerge(t, e))
                merged.erase(k);
              continue;
            }

            auto &t = itT->second;
            auto &e = itE->second;
            if (!considerMerge(t, e))
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

        // Back-to-back vector ops need an explicit PIPE_V barrier.
        if (lastPipe == "V" && consPipe == "V")
          insertTsyncVBefore(consumer);

        // Reverse (overwrite) hazards: if this op defines a tile that was last accessed on a different pipe,
        // fence the pipes before overwriting the tile storage.
        if (opcodeDefinesTile(consOpcode)) {
          auto operands = readOperands(consumer);
          if (!operands.empty()) {
            auto dst = stripIndexing(trim(operands[0]));
            auto dstKey = keyForTile(dst);
            auto itPrev = tileState.find(dstKey);
            if (itPrev != tileState.end()) {
              auto &prev = itPrev->second;
              if (prev.def.op && !prev.def.pipe.empty() && prev.def.pipe != consPipe &&
                  !prev.waitedPipes.count(consPipe.str())) {
                insertFencePairBefore(consumer, prev.def.pipe, consPipe, prev.def.opEnum, consEnum);
                prev.waitedPipes.insert(consPipe.str());
              }
            }
          }
        }

        // Insert fences for cross-pipe RAW tile dependencies (producer -> consumer).
        // Insert record+wait as a pair immediately before the consumer so it is balanced on all dynamic paths.
        {
          std::set<std::pair<std::string, std::string>> insertedPairs;
          for (auto &useSym : opcodeTileUses(consOpcode, consumer, argTypes)) {
            auto use = stripIndexing(trim(useSym));
            auto useKey = keyForTile(use);
            auto defIt = tileState.find(useKey);
            if (defIt == tileState.end())
              defIt = tileState.emplace(useKey, TileSyncState{}).first;
            auto &defState = defIt->second;
            auto &def = defState.def;
            if (def.op && !def.pipe.empty() && def.pipe != consPipe) {
              if (defState.waitedPipes.count(consPipe.str()))
                continue;
              auto k = std::make_pair(def.pipe, consPipe.str());
              if (!insertedPairs.count(k)) {
                insertFencePairBefore(consumer, def.pipe, consPipe, def.opEnum, consEnum);
                insertedPairs.insert(k);
              }
              defState.waitedPipes.insert(consPipe.str());
            }
          }
        }

        // Track last-access pipe for tile operands (RAR/WAR hazards):
        //
        // Many kernels reuse the same tile buffers across ops/iterations. If a tile is *read* on one pipe (e.g. V),
        // then later overwritten by another (e.g. MTE2 via tload), we must fence the pipes to avoid the overwrite
        // clobbering the in-flight read. Record the consumer pipe as the tile's last access so the existing
        // overwrite-hazard logic can insert the required fence when the tile is redefined.
        for (auto &useSym : opcodeTileUses(consOpcode, consumer, argTypes)) {
          auto use = stripIndexing(trim(useSym));
          auto useKey = keyForTile(use);
          auto it = tileState.find(useKey);
          if (it == tileState.end())
            continue;
          TileSyncState st;
          st.def = DefInfo{consumer, consEnum, consPipe.str()};
          st.waitedPipes.insert(consPipe.str());
          it->second = std::move(st);
        }

        // Treat `tstore/tpush(src_tile)` as a pipe access to that tile storage. Later overwrites must wait
        // for the store pipe to finish reading from it (WAR hazard across iterations/tiles).
        if (consOpcode == "tstore" || consOpcode == "tpush" || consOpcode == "mscatter") {
          auto operands = readOperands(consumer);
          std::optional<std::string> src;
          if (consOpcode == "tstore") {
            if (operands.size() >= 2)
              src = stripIndexing(trim(operands.back()));
          } else if (consOpcode == "tpush") {
            // tpush(dst_global, src_tile, token)
            if (operands.size() >= 2)
              src = stripIndexing(trim(operands[1]));
          } else {
            // mscatter(dst_global, src_tile, idx_tile) — treat the first tile operand as the stored tile.
            for (auto &o : operands) {
              auto base = stripIndexing(trim(o));
              if (argTypes.count(base) && isTileTypeString(argTypes.at(base))) {
                src = base;
                break;
              }
            }
          }
          if (src) {
            TileSyncState st;
            st.def = DefInfo{consumer, consEnum, consPipe.str()};
            st.waitedPipes.insert(consPipe.str());
            tileState[keyForTile(*src)] = std::move(st);

	            // A5 correctness: after Acc->GM store, fence FIX -> M so the next matmul cannot
	            // overwrite L0C while the store is still in-flight (see ST `tstore_acc2gm`).
	            if (consEnum == "TSTORE_ACC")
	              insertFencePairAfter(consumer, "FIX", "M", "TSTORE_ACC", "TMATMUL");

	            // A2/A3 correctness: Vec/Mat -> GM stores use PIPE_MTE3, and many kernels reuse the same tile
	            // buffers across loop iterations. Insert a conservative fence so the next iteration's loads
	            // (PIPE_MTE2) cannot overwrite a tile buffer while MTE3 is still reading it.
	            if (consPipe == "MTE3")
	              insertFencePairAfter(consumer, "MTE3", "MTE2", consEnum, "TLOAD");
	          }
	        }

	        // Update last-def for tile results.
	        if (opcodeDefinesTile(consOpcode)) {
	          auto operands = readOperands(consumer);
	          if (!operands.empty()) {
	            auto dst = stripIndexing(trim(operands[0]));
              auto dstKey = keyForTile(dst);
            TileSyncState st;
            st.def = DefInfo{consumer, consEnum, consPipe.str()};
            st.waitedPipes.insert(consPipe.str());
            tileState[dstKey] = std::move(st);
          }
        }

        lastPipe = consPipe.str();
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
