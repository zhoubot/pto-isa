#include "ptobc/canonical_printer.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/AsmState.h>
#include <mlir/IR/Operation.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>

#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ptobc {

static std::vector<std::string> splitLinesPreserveEmpty(const std::string &s) {
  std::vector<std::string> lines;
  std::string cur;
  for (char c : s) {
    if (c == '\n') {
      lines.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  // If the string ended with '\n', cur is empty and represents the final
  // trailing empty line. We do NOT keep that in lines; the join below will
  // re-add newlines.
  lines.push_back(cur);
  return lines;
}

static std::string joinLines(const std::vector<std::string> &lines) {
  std::string out;
  for (size_t i = 0; i < lines.size(); ++i) {
    out += lines[i];
    if (i + 1 < lines.size()) out.push_back('\n');
  }
  return out;
}

static std::string hexFloatLiteral(mlir::FloatAttr a) {
  llvm::SmallString<32> s;
  llvm::raw_svector_ostream os(s);
  llvm::SmallVector<char, 32> digits;
  llvm::APInt bits = a.getValue().bitcastToAPInt();
  bits.toString(digits, /*Radix=*/16, /*Signed=*/false, /*formatAsCLiteral=*/true);
  os << llvm::StringRef(digits.data(), digits.size());
  return os.str().str();
}

static void sortAttributesLexicographically(mlir::ModuleOp module) {
  module.walk([&](mlir::Operation *op) {
    auto attrs = op->getAttrs();
    if (attrs.size() <= 1) return;

    llvm::SmallVector<mlir::NamedAttribute, 8> sorted(attrs.begin(), attrs.end());
    llvm::sort(sorted, [](const mlir::NamedAttribute &a, const mlir::NamedAttribute &b) {
      return a.getName().getValue() < b.getName().getValue();
    });

    // Only write back if something actually changed.
    if (llvm::equal(sorted, attrs, [](const mlir::NamedAttribute &x, const mlir::NamedAttribute &y) {
          return x.getName() == y.getName() && x.getValue() == y.getValue();
        })) {
      return;
    }

    op->setAttrs(sorted);
  });
}

static void canonicalizeScalarFloatConstants(mlir::ModuleOp module,
                                             const mlir::AsmState::LocationMap &locMap,
                                             std::vector<std::string> &lines) {
  // Match: "%name = arith.constant <anything> : fXX"
  // We keep the prefix + the type suffix, replace the literal.
  const std::regex re(R"(^([ \t]*%[-a-zA-Z$._0-9]+[ \t]*=[ \t]*arith\.constant[ \t]+)(.+?)([ \t]*:[ \t]*f(16|32|64)[ \t]*$))");

  module.walk([&](mlir::Operation *op) {
    auto cst = llvm::dyn_cast<mlir::arith::ConstantOp>(op);
    if (!cst) return;

    auto f = llvm::dyn_cast<mlir::FloatAttr>(cst.getValue());
    if (!f) return;

    // Only canonicalize scalar float constants.
    if (!llvm::isa<mlir::FloatType>(cst.getType())) return;

    auto it = locMap.find(op);
    if (it == locMap.end()) return;

    unsigned lineNo = it->second.first;
    if (lineNo == 0) return;
    size_t idx = size_t(lineNo - 1);
    if (idx >= lines.size()) return;

    std::smatch m;
    if (!std::regex_match(lines[idx], m, re)) {
      // If the format doesn't match (e.g., multi-result or dialect-specific
      // printer changes), leave it as-is.
      return;
    }

    std::string lit = hexFloatLiteral(f);
    lines[idx] = m[1].str() + lit + m[3].str();
  });
}

std::string printModuleCanonical(mlir::ModuleOp module,
                                 const CanonicalPrintOptions &opt) {
  // Enforce canonical attribute ordering before printing.
  sortAttributesLexicographically(module);

  mlir::OpPrintingFlags flags;
  flags.useLocalScope();
  flags.assumeVerified();
  if (opt.generic) flags.printGenericOpForm();

  mlir::AsmState::LocationMap locMap;
  mlir::AsmState state(module.getOperation(), flags, &locMap);

  std::string printed;
  llvm::raw_string_ostream os(printed);
  module.getOperation()->print(os, state);
  os.flush();

  if (opt.keepMLIRFloatPrinting) {
    return printed;
  }

  // Canonicalize floats in-place.
  auto lines = splitLinesPreserveEmpty(printed);
  canonicalizeScalarFloatConstants(module, locMap, lines);

  std::string out = joinLines(lines);
  return out;
}

} // namespace ptobc
