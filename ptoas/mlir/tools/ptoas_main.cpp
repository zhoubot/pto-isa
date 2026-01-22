#include "ptoas/BishengDriver.h"
#include "ptoas/CCEmitter.h"
#include "ptoas/PTOASFrontend.h"
#include "ptoas/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <string>

namespace {

static std::string getEnvOrEmpty(const char *name) {
  if (const char *env = std::getenv(name))
    return env;
  return "";
}

} // namespace

int main(int argc, char **argv) {
  llvm::cl::opt<std::string> input(llvm::cl::Positional, llvm::cl::Required, llvm::cl::desc("<input.pto>"));
  llvm::cl::opt<std::string> output("o", llvm::cl::init(""), llvm::cl::desc("Output source path"));
  llvm::cl::opt<std::string> target("target", llvm::cl::init("npu"),
                                    llvm::cl::desc("Target: npu (CCE) or cpu (CPU simulator C++)"));
  llvm::cl::opt<std::string> kernelName("kernel-name", llvm::cl::init("pto_kernel"),
                                        llvm::cl::desc("Generated kernel function name"));
  llvm::cl::opt<std::string> emitBin("emit-bin", llvm::cl::init(""), llvm::cl::desc("Also compile and emit .bin"));
  llvm::cl::opt<std::string> arch("arch", llvm::cl::init("dav-c220-vec"),
                                  llvm::cl::desc("CCE arch (dav-c220-vec/dav-c220-cube/dav-c310)"));
  llvm::cl::opt<std::string> memoryModel("memory-model", llvm::cl::init("MEMORY_BASE"),
                                         llvm::cl::desc("MEMORY_BASE or REGISTER_BASE"));
  llvm::cl::opt<std::string> repoRootOpt("repo-root", llvm::cl::init(""),
                                         llvm::cl::desc("Repo root path (for -I<repo>/include)"));
  llvm::cl::opt<std::string> ascendHomeOpt("ascend-home", llvm::cl::init(""),
                                           llvm::cl::desc("ASCEND_HOME_PATH (for bisheng includes)"));
  llvm::cl::opt<bool> insertEvents("insert-events", llvm::cl::init(false),
                                   llvm::cl::desc("Insert record_event/wait_event for cross-pipe deps (prototype)"));
  llvm::cl::opt<bool> assignTileAddrs("assign-tile-addrs", llvm::cl::init(true),
                                      llvm::cl::desc("Assign default addresses to tiles (prototype)"));

  llvm::cl::ParseCommandLineOptions(argc, argv, "ptoas (MLIR-based prototype)\n");

  mlir::MLIRContext ctx;
  std::string err;
  auto module = ptoas::parsePTOASFile(input, ctx, err);
  if (!module) {
    llvm::errs() << "parse failed: " << err << "\n";
    return 1;
  }

  if (assignTileAddrs || insertEvents) {
    mlir::PassManager pm(&ctx);
    if (assignTileAddrs)
      pm.addPass(ptoas::createAssignTileAddressesPass());
    if (insertEvents)
      pm.addPass(ptoas::createInsertEventsPass());
    if (mlir::failed(pm.run(module))) {
      llvm::errs() << "pass pipeline failed\n";
      return 1;
    }
  }

  // Default output path: <input>.<ext> in CWD.
  std::string outPath = output;
  if (outPath.empty()) {
    llvm::SmallString<256> p(input);
    // NPU sources are still compiled as CCE via `bisheng -xcce`; `.cpp` is used for
    // better editor/tooling compatibility (matches the manual kernels style).
    llvm::sys::path::replace_extension(p, "cpp");
    if (target == "cpu")
      llvm::sys::path::replace_extension(p, "cpu.cpp");
    outPath = p.str().str();
  }

  auto repoRoot = !repoRootOpt.empty() ? repoRootOpt : getEnvOrEmpty("PTO_REPO_ROOT");
  if (repoRoot.empty())
    repoRoot = ".";

  // (NOTE) kernelName is currently a placeholder for future integration; the emitter
  // uses a fixed function name today.
  (void)kernelName;

  std::string outText;
  if (target == "cpu") {
    outText = ptoas::emitCpuCppFromModule(module, repoRoot);
  } else if (target == "npu") {
    outText = ptoas::emitCceFromModule(module, repoRoot, memoryModel);
  } else {
    llvm::errs() << "unknown --target: " << target << " (expected: npu|cpu)\n";
    return 1;
  }
  std::error_code ec;
  llvm::raw_fd_ostream os(outPath, ec, llvm::sys::fs::OF_Text);
  if (ec) {
    llvm::errs() << "failed to write " << outPath << ": " << ec.message() << "\n";
    return 1;
  }
  os << outText;
  os.flush();

  if (!emitBin.empty() && target != "cpu") {
    ptoas::BishengCompileOptions opts;
    opts.ascendHomePath = !ascendHomeOpt.empty() ? ascendHomeOpt : getEnvOrEmpty("ASCEND_HOME_PATH");
    opts.repoRoot = repoRoot;
    opts.arch = arch;
    opts.memoryModel = memoryModel;
    auto err2 = ptoas::compileCceToBin(outPath, emitBin, opts);
    if (!err2.empty()) {
      llvm::errs() << err2 << "\n";
      return 1;
    }
  }

  llvm::outs() << "wrote " << outPath << "\n";
  if (!emitBin.empty() && target != "cpu")
    llvm::outs() << "wrote " << emitBin << "\n";
  return 0;
}
