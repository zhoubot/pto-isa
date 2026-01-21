#include "ptoas/PTOASFrontend.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace ptoas {
namespace {

static std::string stripComment(const std::string &line) {
  auto pos = line.find(';');
  return (pos == std::string::npos) ? line : line.substr(0, pos);
}

static std::string trim(std::string s) {
  auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
  while (!s.empty() && isSpace((unsigned char)s.front()))
    s.erase(s.begin());
  while (!s.empty() && isSpace((unsigned char)s.back()))
    s.pop_back();
  return s;
}

static std::string stripTrailingSemicolon(std::string s) {
  s = trim(std::move(s));
  if (!s.empty() && s.back() == ';') {
    s.pop_back();
    s = trim(std::move(s));
  }
  return s;
}

static std::vector<std::string> splitTopLevelCommas(const std::string &s) {
  std::vector<std::string> out;
  std::string cur;
  int depthParen = 0, depthBrack = 0, depthAngle = 0, depthBrace = 0;

  auto flush = [&]() {
    auto t = trim(cur);
    if (!t.empty())
      out.push_back(std::move(t));
    cur.clear();
  };

  for (char ch : s) {
    switch (ch) {
    case '(':
      depthParen++;
      break;
    case ')':
      depthParen = std::max(0, depthParen - 1);
      break;
    case '[':
      depthBrack++;
      break;
    case ']':
      depthBrack = std::max(0, depthBrack - 1);
      break;
    case '<':
      depthAngle++;
      break;
    case '>':
      depthAngle = std::max(0, depthAngle - 1);
      break;
    case '{':
      depthBrace++;
      break;
    case '}':
      depthBrace = std::max(0, depthBrace - 1);
      break;
    default:
      break;
    }

    if (ch == ',' && depthParen == 0 && depthBrack == 0 && depthAngle == 0 && depthBrace == 0) {
      flush();
      continue;
    }
    cur.push_back(ch);
  }
  flush();
  return out;
}

static std::pair<std::string, std::string> splitOnceOrDie(const std::string &s, char sep, const char *msg) {
  auto pos = s.find(sep);
  if (pos == std::string::npos)
    llvm::report_fatal_error(msg);
  return {trim(s.substr(0, pos)), trim(s.substr(pos + 1))};
}

static void splitAttrAndTypeSig(std::string rest, std::string &operandsPart, std::string &attrDict,
                                std::string &typeSig) {
  operandsPart = trim(rest);
  attrDict.clear();
  typeSig.clear();

  auto bracePos = operandsPart.find('{');
  if (bracePos != std::string::npos) {
    int depth = 0;
    size_t end = std::string::npos;
    for (size_t i = bracePos; i < operandsPart.size(); ++i) {
      if (operandsPart[i] == '{')
        depth++;
      else if (operandsPart[i] == '}') {
        depth--;
        if (depth == 0) {
          end = i;
          break;
        }
      }
    }
    if (end == std::string::npos)
      llvm::report_fatal_error("Unclosed attr dict");
    attrDict = trim(operandsPart.substr(bracePos, end - bracePos + 1));
    operandsPart = trim(operandsPart.substr(0, bracePos) + " " + operandsPart.substr(end + 1));
  }

  auto colonPos = operandsPart.find(':');
  if (colonPos != std::string::npos) {
    typeSig = trim(operandsPart.substr(colonPos + 1));
    operandsPart = trim(operandsPart.substr(0, colonPos));
  }
}

static std::string normalizePtoMnemonic(std::string op) {
  op = trim(std::move(op));
  if (op.rfind("pto.", 0) == 0)
    op = op.substr(4);
  return op;
}

} // namespace

mlir::ModuleOp parsePTOASFile(const std::string &path, mlir::MLIRContext &ctx, std::string &errorOut) {
  errorOut.clear();
  ctx.allowUnregisteredDialects();

  auto bufOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufOrErr) {
    errorOut = "failed to read file: " + path;
    return {};
  }
  std::string text = (*bufOrErr)->getBuffer().str();

  mlir::OpBuilder b(&ctx);
  auto module = mlir::ModuleOp::create(b.getUnknownLoc());
  b.setInsertionPointToStart(module.getBody());

  struct ControlFrame {
    enum Kind { For, If } kind;
    mlir::Operation *op = nullptr;
    bool elseOpened = false;
  };
  std::vector<mlir::Block *> blockStack;
  std::vector<ControlFrame> ctrlStack;
  blockStack.push_back(module.getBody());
  b.setInsertionPointToEnd(blockStack.back());

  auto curBlock = [&]() -> mlir::Block * { return blockStack.back(); };

  auto makeStringArrayAttr = [&](const std::vector<std::string> &items) -> mlir::ArrayAttr {
    llvm::SmallVector<mlir::Attribute> attrs;
    attrs.reserve(items.size());
    for (auto &s : items)
      attrs.push_back(b.getStringAttr(s));
    return b.getArrayAttr(attrs);
  };

  auto parseScfForHeader = [&](const std::string &line, std::string &iv, std::string &lb, std::string &ub,
                               std::string &step) {
    // Syntax:
    //   scf.for %iv = <lb> to <ub> step <step>
    auto rest = trim(line.substr(std::string("scf.for").size()));
    auto parts = splitOnceOrDie(rest, '=', "invalid scf.for (expected: scf.for %iv = lb to ub step step)");
    iv = trim(parts.first);
    auto rhs = trim(parts.second);
    auto toPos = rhs.find(" to ");
    if (toPos == std::string::npos)
      llvm::report_fatal_error("invalid scf.for (missing ' to ')");
    lb = trim(rhs.substr(0, toPos));
    auto rhs2 = trim(rhs.substr(toPos + 4));
    auto stepPos = rhs2.find(" step ");
    if (stepPos == std::string::npos)
      llvm::report_fatal_error("invalid scf.for (missing ' step ')");
    ub = trim(rhs2.substr(0, stepPos));
    step = trim(rhs2.substr(stepPos + 6));
  };

  auto parseScfIfHeader = [&](const std::string &line, std::string &cond) {
    // Syntax:
    //   scf.if <cond>
    cond = trim(line.substr(std::string("scf.if").size()));
    if (cond.empty())
      llvm::report_fatal_error("invalid scf.if (expected: scf.if %cond)");
  };

  size_t lineNo = 0;
  size_t cur = 0;
  while (cur <= text.size()) {
    size_t next = text.find('\n', cur);
    if (next == std::string::npos)
      next = text.size();
    std::string raw = text.substr(cur, next - cur);
    cur = next + 1;
    lineNo++;

    std::string line = stripTrailingSemicolon(stripComment(raw));
    line = trim(std::move(line));
    if (line.empty())
      continue;

    auto loc = b.getUnknownLoc();

    // Control-flow braces.
    if (line == "}" || line == "} else {") {
      if (blockStack.size() <= 1 || ctrlStack.empty())
        llvm::report_fatal_error("unmatched '}'");

      if (line == "} else {") {
        auto &top = ctrlStack.back();
        if (top.kind != ControlFrame::If)
          llvm::report_fatal_error("'} else {' only valid for scf.if");
        if (top.elseOpened)
          llvm::report_fatal_error("duplicate else for scf.if");
        top.elseOpened = true;

        // Switch from then to else region.
        blockStack.pop_back();
        auto &elseRegion = top.op->getRegion(1);
        if (elseRegion.empty())
          elseRegion.push_back(new mlir::Block());
        blockStack.push_back(&elseRegion.front());
        b.setInsertionPointToEnd(curBlock());
        continue;
      }

      // Close the current region and return to parent block.
      blockStack.pop_back();
      ctrlStack.pop_back();
      b.setInsertionPointToEnd(curBlock());
      continue;
    }

    // scf.for / scf.if block openers.
    if ((line.rfind("scf.for", 0) == 0 || line.rfind("scf.if", 0) == 0) && !line.empty() && line.back() == '{') {
      std::string header = trim(line.substr(0, line.size() - 1)); // drop trailing '{'
      if (header.rfind("scf.for", 0) == 0) {
        std::string iv, lb, ub, step;
        parseScfForHeader(header, iv, lb, ub, step);
        mlir::OperationState st(loc, "scf.for");
        st.addAttribute("operands", makeStringArrayAttr({iv, lb, ub, step}));
        st.addRegion();
        auto *forOp = b.create(st);
        auto &r = forOp->getRegion(0);
        r.push_back(new mlir::Block());
        ctrlStack.push_back({ControlFrame::For, forOp, false});
        blockStack.push_back(&r.front());
        b.setInsertionPointToEnd(curBlock());
        continue;
      }
      if (header.rfind("scf.if", 0) == 0) {
        std::string cond;
        parseScfIfHeader(header, cond);
        mlir::OperationState st(loc, "scf.if");
        st.addAttribute("operands", makeStringArrayAttr({cond}));
        // Create both then/else regions up front; else may stay empty.
        st.addRegion();
        st.addRegion();
        auto *ifOp = b.create(st);
        auto &thenR = ifOp->getRegion(0);
        thenR.push_back(new mlir::Block());
        ctrlStack.push_back({ControlFrame::If, ifOp, false});
        blockStack.push_back(&thenR.front());
        b.setInsertionPointToEnd(curBlock());
        continue;
      }
    }

    if (line.rfind(".arg ", 0) == 0) {
      if (blockStack.size() != 1)
        llvm::report_fatal_error(".arg must appear at top-level (outside scf regions)");
      auto rest = trim(line.substr(5));
      auto parts = splitOnceOrDie(rest, ':', "invalid .arg (expected: .arg %name : type)");
      auto name = parts.first;
      auto typeStr = parts.second;
      mlir::OperationState st(loc, "pto.arg");
      st.addAttribute("name", b.getStringAttr(name));
      st.addAttribute("type", b.getStringAttr(typeStr));
      b.create(st);
      continue;
    }

    if (line.rfind(".const ", 0) == 0) {
      if (blockStack.size() != 1)
        llvm::report_fatal_error(".const must appear at top-level (outside scf regions)");
      auto rest = trim(line.substr(7));
      auto parts = splitOnceOrDie(rest, ':', "invalid .const (expected: .const %name = lit : type)");
      auto lhs = parts.first;
      auto typeStr = parts.second;
      auto parts2 = splitOnceOrDie(lhs, '=', "invalid .const (expected '=' in lhs)");
      auto name = parts2.first;
      auto value = parts2.second;
      mlir::OperationState st(loc, "pto.const");
      st.addAttribute("name", b.getStringAttr(name));
      st.addAttribute("value", b.getStringAttr(value));
      st.addAttribute("type", b.getStringAttr(typeStr));
      b.create(st);
      continue;
    }

    // SSA-style destination binding (PTO-AS sugar):
    //
    //   %dst = pto.tadd %src0, %src1 : ...
    //
    // Lowers to the existing DPS-like internal form by inserting %dst as the first operand:
    //
    //   pto.tadd operands = ["%dst", "%src0", "%src1"]
    //
    // Also used for declaration-like helpers:
    //   %t0 = pto.alloc_tile %addr : !pto.tile<...>
    //   %x  = pto.make_tensor_view %arg0, ... : !pto.tensor<...>
    auto eqPos = line.find('=');
    if (eqPos != std::string::npos && line.rfind(".const ", 0) != 0 && line.rfind(".arg ", 0) != 0) {
      auto lhs = trim(line.substr(0, eqPos));
      auto rhs = trim(line.substr(eqPos + 1));
      if (lhs.empty() || rhs.empty())
        llvm::report_fatal_error("invalid assignment (expected: %dst = <op> ...)");

      std::string opcode;
      std::string rest;
      auto sp = rhs.find(' ');
      if (sp == std::string::npos) {
        opcode = trim(rhs);
        rest = "";
      } else {
        opcode = trim(rhs.substr(0, sp));
        rest = trim(rhs.substr(sp + 1));
      }

      opcode = normalizePtoMnemonic(opcode);
      std::string operandsPart, attrDict, typeSig;
      splitAttrAndTypeSig(rest, operandsPart, attrDict, typeSig);

      auto rhsOperands = splitTopLevelCommas(operandsPart);
      llvm::SmallVector<mlir::Attribute> operandAttrs;
      operandAttrs.reserve(1 + rhsOperands.size());
      operandAttrs.push_back(b.getStringAttr(lhs));
      for (auto &opnd : rhsOperands)
        operandAttrs.push_back(b.getStringAttr(opnd));

      mlir::OperationState st(loc, ("pto." + opcode).c_str());
      st.addAttribute("operands", b.getArrayAttr(operandAttrs));
      if (!attrDict.empty())
        st.addAttribute("attrs", b.getStringAttr(attrDict));
      if (!typeSig.empty())
        st.addAttribute("typesig", b.getStringAttr(typeSig));
      b.create(st);
      continue;
    }

    // Instruction: <opcode> [<operand_list>] [attr_dict] [: type_sig]
    //
    // For convenience (and to support "marker" statements like `prologue`),
    // we also accept opcode-only lines with no operands.
    std::string opcode;
    std::string rest;
    auto space = line.find(' ');
    if (space == std::string::npos) {
      opcode = trim(line);
      rest = "";
    } else {
      opcode = trim(line.substr(0, space));
      rest = trim(line.substr(space + 1));
    }
    opcode = normalizePtoMnemonic(opcode);
    std::string operandsPart, attrDict, typeSig;
    splitAttrAndTypeSig(rest, operandsPart, attrDict, typeSig);

    auto operands = splitTopLevelCommas(operandsPart);
    llvm::SmallVector<mlir::Attribute> operandAttrs;
    operandAttrs.reserve(operands.size());
    for (auto &op : operands)
      operandAttrs.push_back(b.getStringAttr(op));

    mlir::OperationState st(loc, ("pto." + opcode).c_str());
    st.addAttribute("operands", b.getArrayAttr(operandAttrs));
    if (!attrDict.empty())
      st.addAttribute("attrs", b.getStringAttr(attrDict));
    if (!typeSig.empty())
      st.addAttribute("typesig", b.getStringAttr(typeSig));
    b.create(st);
  }

  if (!ctrlStack.empty())
    llvm::report_fatal_error("unclosed scf region (missing '}')");

  return module;
}

} // namespace ptoas
