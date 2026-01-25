//===---------------- SplitKernelsPass.cpp --------------------------------===//
//
// Prototype multi-kernel splitter for PTO-AS modules.
//
// Motivation:
// - Some workloads (e.g. FlashAttention) are easier to debug/compose as multiple
//   sequential kernels (e.g. cube matmul stage, vector softmax stage, ...).
// - The Python frontend can emit simple marker ops like:
//     pto.stage_qk
//     pto.stage_softmax
//     pto.stage_pv
// - This pass removes these marker ops and annotates subsequent ops with a
//   `ptoas.stage` StringAttr so codegen can emit one kernel per stage.
//
// Notes:
// - This pass is intentionally minimal and only operates on the unregistered
//   `pto.*` ops emitted by PTOASFrontend.cpp.
// - No CFG restructuring is performed; stage markers are expected to appear at
//   top-level only (not inside scf regions).
//===----------------------------------------------------------------------===//

#include "ptoas/Passes.h"
#include "ptoas/ProtoAttrs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

#include <cctype>
#include <string>
#include <vector>

namespace ptoas {
namespace {

static bool isStageMarker(llvm::StringRef opName) { return opName.starts_with("pto.stage_"); }

static std::string stageFromMarker(llvm::StringRef opName) {
  // `pto.stage_<name>` -> "<name>"
  auto s = opName.drop_front(std::string("pto.stage_").size()).str();
  if (s.empty())
    s = "stage";
  // Keep stage names identifier-friendly so later passes can use them as attribute keys.
  std::string out;
  out.reserve(s.size());
  for (char ch : s) {
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_')
      out.push_back(ch);
    else
      out.push_back('_');
  }
  if (!out.empty() && std::isdigit(static_cast<unsigned char>(out.front())))
    out.insert(out.begin(), '_');
  return out.empty() ? std::string("stage") : out;
}

static void annotateNestedRegions(mlir::Operation &op, mlir::StringAttr stageAttr) {
  for (auto &r : op.getRegions()) {
    for (auto &b : r) {
      for (auto &nested : b.getOperations()) {
        auto n = nested.getName().getStringRef();
        if (isStageMarker(n)) {
          llvm::report_fatal_error("pto.stage_* markers are only supported at top-level (not inside scf regions)");
        }
        nested.setAttr(kStageAttr, stageAttr);
        annotateNestedRegions(nested, stageAttr);
      }
    }
  }
}

struct SplitKernelsPass : public mlir::PassWrapper<SplitKernelsPass, mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitKernelsPass)

  llvm::StringRef getArgument() const final { return "ptoas-split-kernels"; }
  llvm::StringRef getDescription() const final {
    return "Split PTO-AS into multiple kernels using `pto.stage_*` markers (prototype)";
  }

  void runOnOperation() override {
    auto module = getOperation();

    bool sawMarker = false;
    std::string currentStage = kPreambleStage;
    std::vector<mlir::Operation *> toErase;

    for (auto &op : module.getBody()->getOperations()) {
      auto name = op.getName().getStringRef();
      if (isStageMarker(name)) {
        sawMarker = true;
        currentStage = stageFromMarker(name);
        toErase.push_back(&op);
        continue;
      }

      // If we never saw a marker, keep the module unchanged.
      if (!sawMarker)
        continue;

      auto stageA = mlir::StringAttr::get(op.getContext(), currentStage);
      op.setAttr(kStageAttr, stageA);
      annotateNestedRegions(op, stageA);
    }

    if (!sawMarker)
      return;

    for (auto *op : toErase)
      op->erase();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createSplitKernelsPass() { return std::make_unique<SplitKernelsPass>(); }

} // namespace ptoas
