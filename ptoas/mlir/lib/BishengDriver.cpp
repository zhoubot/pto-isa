#include "ptoas/BishengDriver.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace ptoas {
namespace {

static bool pathExists(const std::string &p) {
  return llvm::sys::fs::exists(p);
}

static std::vector<std::string> ascendIncludeDirs(const std::string &ascendHome) {
  // Mirrors the include chain needed by `kernel_operator.h` + PTO headers.
  std::vector<std::string> candidates = {
      ascendHome + "/compiler/ascendc/include/basic_api",
      ascendHome + "/compiler/ascendc/include/basic_api/impl",
      ascendHome + "/compiler/asc/include/basic_api",
      ascendHome + "/compiler/asc/include/interface",
      ascendHome + "/compiler/asc",
      ascendHome + "/include/ascendc",
      ascendHome + "/include",
      ascendHome + "/runtime/include",
  };
  std::vector<std::string> out;
  for (auto &p : candidates)
    if (pathExists(p))
      out.push_back(p);
  return out;
}

static std::string shellEscape(const std::string &s) {
  std::ostringstream os;
  os << '\'';
  for (char c : s) {
    if (c == '\'')
      os << "'\\''";
    else
      os << c;
  }
  os << '\'';
  return os.str();
}

static int runOrErr(const std::vector<std::string> &argv, std::string &err) {
  if (argv.empty()) {
    err = "empty argv";
    return 1;
  }
  std::ostringstream cmd;
  for (size_t i = 0; i < argv.size(); ++i) {
    if (i)
      cmd << " ";
    cmd << shellEscape(argv[i]);
  }
  int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    err = "command failed (rc=" + std::to_string(rc) + "): " + cmd.str();
  }
  return rc;
}

} // namespace

std::string compileCceToBin(const std::string &ccePath, const std::string &outBinPath,
                            const BishengCompileOptions &opts) {
  if (opts.ascendHomePath.empty())
    return "ASCEND_HOME_PATH is empty";
  if (opts.repoRoot.empty())
    return "repoRoot is empty";
  if (opts.arch.empty())
    return "arch is empty";
  if (opts.memoryModel.empty())
    return "memoryModel is empty";

  std::string objPath = outBinPath + ".o";

  std::vector<std::string> bishengCmd;
  bishengCmd.push_back("bisheng");
  bishengCmd.push_back("-xcce");
  bishengCmd.push_back("--cce-aicore-arch=" + opts.arch);
  bishengCmd.push_back("-std=c++17");
  // The emitted `.cce` includes `#define MEMORY_BASE` / `#define REGISTER_BASE`,
  // so we don't need (and should avoid) duplicating it on the command line.

  for (auto &inc : ascendIncludeDirs(opts.ascendHomePath))
    bishengCmd.push_back("-I" + inc);
  bishengCmd.push_back("-I" + opts.repoRoot + "/include");

  bishengCmd.push_back("-c");
  bishengCmd.push_back(ccePath);
  bishengCmd.push_back("-o");
  bishengCmd.push_back(objPath);

  std::string err;
  if (runOrErr(bishengCmd, err) != 0) {
    return "bisheng failed: " + err;
  }

  std::vector<std::string> objcopyCmd = {
      "objcopy",
      "--dump-section",
      "__aicore_rel_binary=" + outBinPath,
      objPath,
  };
  if (runOrErr(objcopyCmd, err) != 0) {
    return "objcopy failed: " + err;
  }
  return "";
}

} // namespace ptoas
