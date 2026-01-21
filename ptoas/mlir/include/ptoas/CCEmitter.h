#pragma once

#include <string>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace ptoas {

// Emits an Ascend kernel CCE source file for a parsed PTO-AS module.
// The module is expected to contain PTO-AS "ops" that carry operands/types
// as string attributes (see PTOASFrontend).
std::string emitCceFromModule(mlir::ModuleOp module, const std::string &repoRoot,
                              const std::string &memoryModel);

// Emits a CPU-simulator C++ source file that uses the PTO CPU backend
// (`-D__CPU_SIM`) and runs synchronously (no events/tsync needed).
std::string emitCpuCppFromModule(mlir::ModuleOp module, const std::string &repoRoot);

} // namespace ptoas
