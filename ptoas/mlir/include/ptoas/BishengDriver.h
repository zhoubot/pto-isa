#pragma once

#include <string>

namespace ptoas {

struct BishengCompileOptions {
  std::string ascendHomePath; // ASCEND_HOME_PATH
  std::string repoRoot;       // pto-isa repo root (for include/pto)
  std::string arch;           // dav-c220-vec / dav-c220-cube / dav-c310
  std::string memoryModel;    // MEMORY_BASE / REGISTER_BASE
};

// Compiles a generated CCE source into an object and dumps `__aicore_rel_binary`
// section into `outBinPath`.
//
// Returns empty string on success; otherwise returns a human-readable error.
std::string compileCceToBin(const std::string &ccePath, const std::string &outBinPath,
                            const BishengCompileOptions &opts);

} // namespace ptoas

