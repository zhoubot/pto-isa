#include "ptoas/Passes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace ptoas {
namespace {

static llvm::StringRef stripDialect(llvm::StringRef opName) {
  if (auto dot = opName.find('.'); dot != llvm::StringRef::npos)
    return opName.drop_front(dot + 1);
  return opName;
}

static bool isPtoMetaOp(mlir::Operation *op) {
  auto name = op->getName().getStringRef();
  return name == "pto.arg" || name == "pto.const" || name == "pto.record_event" || name == "pto.tsync" ||
         name == "pto.make_tensor_view" || name == "pto.alloc_tile";
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

struct InsertEventsPass : public mlir::PassWrapper<InsertEventsPass, mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    auto *block = module.getBody();
    if (!block)
      return;

    // Collect `.arg` types so we can choose TMOV/TSTORE enum variants correctly.
    std::map<std::string, std::string> argTypes;
    for (auto &op : block->getOperations()) {
      auto name = op.getName().getStringRef();
      if (name == "pto.arg") {
        auto n = op.getAttrOfType<mlir::StringAttr>("name");
        auto t = op.getAttrOfType<mlir::StringAttr>("type");
        if (!n || !t)
          continue;
        argTypes[n.getValue().str()] = t.getValue().str();
        continue;
      }
      if (name == "pto.alloc_tile") {
        auto operands = readOperands(&op);
        if (operands.empty())
          continue;
        auto typeSig = op.getAttrOfType<mlir::StringAttr>("typesig");
        if (!typeSig)
          continue;
        argTypes[operands[0]] = typeSig.getValue().str();
        continue;
      }
    }

    mlir::OpBuilder b(module.getContext());
    int nextEventId = 0;

    for (auto it = block->begin(); it != block->end(); ++it) {
      mlir::Operation *producer = &*it;
      if (!isPtoInstrOp(producer))
        continue;

      auto nextIt = std::next(it);
      if (nextIt == block->end())
        break;

      // Skip if there's already a record_event right after producer; we assume pass already ran.
      if (nextIt->getName().getStringRef() == "pto.record_event")
        continue;

      // Find the next instruction (skip meta ops that might already exist).
      auto consumerIt = nextIt;
      while (consumerIt != block->end() && !isPtoInstrOp(&*consumerIt)) {
        ++consumerIt;
      }
      if (consumerIt == block->end())
        break;
      mlir::Operation *consumer = &*consumerIt;

      auto prodOpcode = stripDialect(producer->getName().getStringRef());
      auto consOpcode = stripDialect(consumer->getName().getStringRef());

      auto srcEnum = opcodeToOpEnum(prodOpcode, producer, argTypes);
      auto dstEnum = opcodeToOpEnum(consOpcode, consumer, argTypes);
      if (srcEnum.empty() || dstEnum.empty())
        continue;
      auto srcPipe = pipeForOpEnum(srcEnum);
      auto dstPipe = pipeForOpEnum(dstEnum);
      if (srcPipe.empty() || dstPipe.empty())
        continue;
      if (srcPipe == dstPipe)
        continue;

      std::string eventName = ("e" + std::to_string(nextEventId++));

      // Insert record_event after producer.
      b.setInsertionPointAfter(producer);
      {
        mlir::OperationState st(producer->getLoc(), "pto.record_event");
        st.addAttribute("name", b.getStringAttr(eventName));
        st.addAttribute("src", b.getStringAttr(srcEnum));
        st.addAttribute("dst", b.getStringAttr(dstEnum));
        b.create(st);
      }

      // Insert tsync immediately before consumer (unless there's already one there).
      auto *prev = consumer->getPrevNode();
      if (!prev || prev->getName().getStringRef() != "pto.tsync") {
        b.setInsertionPoint(consumer);
        mlir::OperationState st(consumer->getLoc(), "pto.tsync");
        st.addAttribute("events", b.getArrayAttr({b.getStringAttr(eventName)}));
        b.create(st);
      }
    }
  }

  llvm::StringRef getArgument() const final { return "ptoas-insert-events"; }
  llvm::StringRef getDescription() const final {
    return "Insert tsync + record_event between memory/vector pipeline ops (prototype).";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertEventsPass() {
  return std::make_unique<InsertEventsPass>();
}

} // namespace ptoas
