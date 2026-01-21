#include "ptoas/CCEmitter.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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

static std::string mnemonicFor(llvm::StringRef opName) {
  // PTO-AS frontend builds most instructions as unregistered `pto.<mnemonic>` ops.
  // For control-flow, we also allow unregistered `scf.*` ops (to match MLIR names).
  if (opName.starts_with("pto."))
    return opName.drop_front(4).str();
  return opName.str();
}

static std::string elemToCpp(const std::string &dtype) {
  if (dtype == "f16" || dtype == "bf16")
    return "half";
  if (dtype == "f32")
    return "float";
  if (dtype == "i32")
    return "int32_t";
  if (dtype == "u32")
    return "uint32_t";
  llvm::report_fatal_error(llvm::Twine("Unsupported dtype: ") + dtype);
}

static std::string intOrDynamic(const std::string &v) {
  if (v == "dyn")
    return "pto::DYNAMIC";
  return v;
}

static std::vector<std::string> defaultStrideForShape5(const std::vector<std::string> &shape5) {
  // PTO-AS canonical default uses a simple 2D-view stride:
  //   stride=[1,1,1,<cols>,1]
  // This matches docs/grammar/PTO-AS.md and is sufficient when dims [0..2] are 1.
  return { "1", "1", "1", shape5.at(4), "1" };
}

static std::vector<std::string> parseList5(const std::string &s) {
  auto t = trim(s);
  if (t.size() < 2 || t.front() != '[' || t.back() != ']')
    llvm::report_fatal_error(llvm::Twine("Expected list literal [..]: ") + t);
  std::vector<std::string> items;
  std::string inner = t.substr(1, t.size() - 2);
  for (auto &p : splitTopLevelCommas(inner))
    items.push_back(trim(p));
  if (items.size() != 5)
    llvm::report_fatal_error(llvm::Twine("Expected 5 elements: ") + t);
  return items;
}

static std::optional<std::map<std::string, std::string>> tryParseAngleKVs(const std::string &typeStr,
                                                                          llvm::StringRef prefix) {
  auto t = trim(typeStr);
  if (!llvm::StringRef(t).starts_with(prefix))
    return std::nullopt;
  auto l = t.find('<');
  auto r = t.rfind('>');
  if (l == std::string::npos || r == std::string::npos || r <= l)
    llvm::report_fatal_error(llvm::Twine("Malformed type: ") + t);
  std::string inner = t.substr(l + 1, r - l - 1);
  std::map<std::string, std::string> kv;
  for (auto &p : splitTopLevelCommas(inner)) {
    auto eq = p.find('=');
    if (eq == std::string::npos)
      continue;
    kv[trim(p.substr(0, eq))] = trim(p.substr(eq + 1));
  }
  return kv;
}

struct ArgInfo {
  std::string name;    // with leading %
  std::string typeStr; // PTO-AS type string
};

struct ConstInfo {
  std::string name;  // with leading %
  std::string value; // literal text
  std::string type;  // type text
};

struct RecordEventInfo {
  std::string name; // e0
  std::string src;  // Op enum string
  std::string dst;  // Op enum string
};

static bool isTileType(const std::string &typeStr) { return llvm::StringRef(typeStr).starts_with("!pto.tile<"); }

static bool isTensorType(const std::string &typeStr) {
  auto s = llvm::StringRef(typeStr);
  return s.starts_with("!pto.tensor<") || s.starts_with("!pto.gtensor<");
}

static RecordEventInfo readRecordEvent(mlir::Operation *op) {
  auto name = op->getAttrOfType<mlir::StringAttr>("name");
  auto src = op->getAttrOfType<mlir::StringAttr>("src");
  auto dst = op->getAttrOfType<mlir::StringAttr>("dst");
  if (!name || !src || !dst)
    llvm::report_fatal_error("pto.record_event missing attrs");
  return {name.getValue().str(), src.getValue().str(), dst.getValue().str()};
}

static std::vector<std::string> readOperands(mlir::Operation *op) {
  auto arr = op->getAttrOfType<mlir::ArrayAttr>("operands");
  if (!arr)
    llvm::report_fatal_error(llvm::Twine(op->getName().getStringRef()) + ": missing operands attr");
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

static std::pair<std::string, std::optional<std::pair<std::string, std::string>>> parseIndexedOperand(
    const std::string &s) {
  auto l = s.find('[');
  auto r = s.rfind(']');
  if (l == std::string::npos || r == std::string::npos || r <= l)
    return {trim(s), std::nullopt};
  std::string base = trim(s.substr(0, l));
  std::string inside = trim(s.substr(l + 1, r - l - 1));
  auto parts = splitTopLevelCommas(inside);
  if (parts.size() != 2)
    llvm::report_fatal_error(llvm::Twine("expected 2 indices in ") + s);
  return {base, std::make_pair(trim(parts[0]), trim(parts[1]))};
}

static std::string cppLayout(const std::string &v) { return "Layout::" + v; }
static std::string cppTileType(const std::string &v) { return "TileType::" + v; }
static std::string cppBLayout(const std::string &v) { return "BLayout::" + v; }
static std::string cppSLayout(const std::string &v) { return "SLayout::" + v; }
static std::string cppPad(const std::string &v) { return "PadValue::" + v; }

static std::string defaultFractalBytesForTileLoc(const std::string &loc) {
  if (loc == "Acc")
    return "1024";
  return "512";
}

static std::string indentExtra(const std::string &s, const std::string &extra) {
  if (extra.empty())
    return s;
  std::string out;
  out.reserve(s.size() + extra.size() * 4);
  out += extra;
  for (size_t i = 0; i < s.size(); ++i) {
    out.push_back(s[i]);
    if (s[i] == '\n' && i + 1 < s.size())
      out += extra;
  }
  return out;
}

} // namespace

std::string emitCceFromModule(mlir::ModuleOp module, const std::string &repoRoot, const std::string &memoryModel) {
  std::vector<ArgInfo> args;
  std::vector<ConstInfo> consts;
  std::vector<RecordEventInfo> events;

  for (auto &op : module.getBody()->getOperations()) {
    auto name = op.getName().getStringRef();
    if (name == "pto.arg") {
      auto n = op.getAttrOfType<mlir::StringAttr>("name");
      auto t = op.getAttrOfType<mlir::StringAttr>("type");
      if (!n || !t)
        llvm::report_fatal_error("pto.arg missing attrs");
      args.push_back({n.getValue().str(), t.getValue().str()});
      continue;
    }
    if (name == "pto.const") {
      auto n = op.getAttrOfType<mlir::StringAttr>("name");
      auto v = op.getAttrOfType<mlir::StringAttr>("value");
      auto t = op.getAttrOfType<mlir::StringAttr>("type");
      if (!n || !v || !t)
        llvm::report_fatal_error("pto.const missing attrs");
      consts.push_back({n.getValue().str(), v.getValue().str(), t.getValue().str()});
      continue;
    }
  }

  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "pto.record_event")
      events.push_back(readRecordEvent(op));
  });

  // Classify args.
  struct TensorArg {
    ArgInfo a;
    std::map<std::string, std::string> kv;
  };
  struct TileLocal {
    ArgInfo a;
    std::map<std::string, std::string> kv;
  };

  std::vector<TensorArg> tensorArgs;
  std::vector<TileLocal> tileLocals;
  for (auto &a : args) {
    if (isTensorType(a.typeStr)) {
      auto kvOpt = tryParseAngleKVs(a.typeStr, "!pto.tensor");
      if (!kvOpt)
        kvOpt = tryParseAngleKVs(a.typeStr, "!pto.gtensor"); // compat
      if (!kvOpt)
        llvm::report_fatal_error(llvm::Twine("tensor type parse failed: ") + a.typeStr);
      auto kv = *kvOpt;
      if (kv.find("dtype") == kv.end() && kv.find("element") != kv.end())
        kv["dtype"] = kv["element"];
      tensorArgs.push_back({a, std::move(kv)});
    } else if (isTileType(a.typeStr)) {
      auto kvOpt = tryParseAngleKVs(a.typeStr, "!pto.tile");
      if (!kvOpt)
        llvm::report_fatal_error(llvm::Twine("tile type parse failed: ") + a.typeStr);
      auto kv = *kvOpt;
      if (kv.find("dtype") == kv.end() && kv.find("element") != kv.end())
        kv["dtype"] = kv["element"];
      tileLocals.push_back({a, std::move(kv)});
    }
  }

  // Emit.
  std::ostringstream os;
  os << "// Generated by ptoas (mlir prototype)\n";
  os << "#define " << memoryModel << "\n";
  os << "#include \"kernel_operator.h\"\n";
  os << "#include <pto/pto-inst.hpp>\n";
  os << "#include <cstdint>\n";
  os << "using namespace pto;\n\n";

  // Kernel signature: tensors only.
  os << "extern \"C\" __global__ AICORE void pto_kernel(";
  for (size_t i = 0; i < tensorArgs.size(); ++i) {
    if (i)
      os << ", ";
    os << "GM_ADDR " << tensorArgs[i].a.name.substr(1);
  }
  os << ") {\n";

  // Tensor objects.
  for (auto &t : tensorArgs) {
    auto &kv = t.kv;
    auto dtype = kv.at("dtype");
    auto shape = parseList5(kv.at("shape"));
    auto stride = kv.count("stride") ? parseList5(kv.at("stride")) : defaultStrideForShape5(shape);
    auto layout = kv.count("layout") ? kv.at("layout") : "ND";
    auto elemCpp = elemToCpp(dtype);
    auto baseName = t.a.name.substr(1);
    os << "  using " << baseName << "_Shape = Shape<" << intOrDynamic(shape[0]) << ", " << intOrDynamic(shape[1])
       << ", " << intOrDynamic(shape[2]) << ", " << intOrDynamic(shape[3]) << ", " << intOrDynamic(shape[4]) << ">;\n";
    os << "  using " << baseName << "_Stride = Stride<" << intOrDynamic(stride[0]) << ", " << intOrDynamic(stride[1])
       << ", " << intOrDynamic(stride[2]) << ", " << intOrDynamic(stride[3]) << ", " << intOrDynamic(stride[4])
       << ">;\n";
    os << "  using " << baseName << "_Tensor = GlobalTensor<" << elemCpp << ", " << baseName << "_Shape, " << baseName
       << "_Stride, " << cppLayout(layout) << ">;\n";
    os << "  " << baseName << "_Tensor g_" << baseName << "((__gm__ " << elemCpp << "*)" << baseName << ");\n\n";
  }

  // Tile locals.
  for (auto &t : tileLocals) {
    auto &kv = t.kv;
    auto dtype = kv.at("dtype");
    auto elemCpp = elemToCpp(dtype);
    auto rows = intOrDynamic(kv.at("rows"));
    auto cols = intOrDynamic(kv.at("cols"));
    auto loc = kv.count("loc") ? kv.at("loc") : "Vec";
    auto blayout = kv.count("blayout") ? kv.at("blayout") : "RowMajor";
    auto slayout = kv.count("slayout") ? kv.at("slayout") : "NoneBox";
    auto fractal = kv.count("fractal") ? kv.at("fractal") : defaultFractalBytesForTileLoc(loc);
    auto pad = kv.count("pad") ? kv.at("pad") : "Null";
    auto valid = kv.count("valid") ? kv.at("valid") : (kv.at("rows") + "x" + kv.at("cols"));
    auto x = valid.find('x');
    if (x == std::string::npos)
      llvm::report_fatal_error("tile valid must be RxC");
    auto vrow = intOrDynamic(valid.substr(0, x));
    auto vcol = intOrDynamic(valid.substr(x + 1));

    auto baseName = t.a.name.substr(1);
    os << "  using " << baseName << "_Tile = Tile<" << cppTileType(loc) << ", " << elemCpp << ", " << rows << ", "
       << cols << ", " << cppBLayout(blayout) << ", " << vrow << ", " << vcol << ", " << cppSLayout(slayout) << ", "
       << intOrDynamic(fractal) << ", " << cppPad(pad) << ">;\n";
    os << "  " << baseName << "_Tile t_" << baseName << ";\n";
  }
  if (!tileLocals.empty())
    os << "\n";

  // Constants.
  for (auto &c : consts) {
    auto name = c.name.substr(1);
    auto v = trim(c.value);
    if (v.rfind("0x", 0) == 0 || v.rfind("0X", 0) == 0)
      os << "  constexpr uint64_t c_" << name << " = " << v << ";\n";
    else
      os << "  constexpr int64_t c_" << name << " = " << v << ";\n";
  }
  if (!consts.empty())
    os << "\n";

  // Declare events.
  for (auto &e : events) {
    os << "  Event<Op::" << e.src << ", Op::" << e.dst << "> " << e.name << ";\n";
  }
  if (!events.empty())
    os << "\n";

  auto resolve = [&](const std::string &v) -> std::string {
    auto t = trim(v);
    if (!t.empty() && t[0] == '%') {
      auto key = t.substr(1);
      // Tensor args become g_<name>, tile locals become t_<name>, consts become c_<name>.
      for (auto &ta : tensorArgs)
        if (ta.a.name.substr(1) == key)
          return "g_" + key;
      for (auto &tl : tileLocals)
        if (tl.a.name.substr(1) == key)
          return "t_" + key;
      for (auto &c : consts)
        if (c.name.substr(1) == key)
          return "c_" + key;
      return key; // allow raw variable names
    }
    return t;
  };

  auto emitInstrCall = [&](mlir::Operation *op, const std::string *assignEvent) -> std::string {
    auto opcode = mnemonicFor(op->getName().getStringRef());
    auto assignPrefix = [&]() -> std::string {
      if (!assignEvent)
        return "  ";
      return "  " + *assignEvent + " = ";
    };

    // Marker/scalar ops (prototype): used to make PTO-AS inputs look like complete kernels.
    if (opcode == "prologue")
      return "  // prologue\n";
    if (opcode == "epilogue")
      return "  // epilogue\n";

    if (opcode == "get_block_idx") {
      auto operands = readOperands(op);
      if (operands.size() != 1)
        llvm::report_fatal_error("get_block_idx expects 1 operand (dest)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  int64_t " + dst + " = static_cast<int64_t>(get_block_idx());\n";
    }

    if (opcode == "get_block_num") {
      auto operands = readOperands(op);
      if (operands.size() != 1)
        llvm::report_fatal_error("get_block_num expects 1 operand (dest)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  int64_t " + dst + " = static_cast<int64_t>(get_block_num());\n";
    }

    if (opcode == "iadd") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("iadd expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") + (" + resolve(operands[2]) + ");\n";
    }

    if (opcode == "imul") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("imul expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") * (" + resolve(operands[2]) + ");\n";
    }

    auto emitIcmp = [&](const char *expect, const char *cxxOp) -> std::optional<std::string> {
      if (opcode != expect)
        return std::nullopt;
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error(llvm::Twine(expect) + " expects 3 operands (dst, lhs, rhs)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") " + cxxOp + " (" + resolve(operands[2]) + ");\n";
    };

    if (auto s = emitIcmp("icmp_eq", "=="))
      return *s;
    if (auto s = emitIcmp("icmp_ne", "!="))
      return *s;
    if (auto s = emitIcmp("icmp_lt", "<"))
      return *s;
    if (auto s = emitIcmp("icmp_le", "<="))
      return *s;
    if (auto s = emitIcmp("icmp_gt", ">"))
      return *s;
    if (auto s = emitIcmp("icmp_ge", ">="))
      return *s;

    if (opcode == "scf.yield")
      return "  // scf.yield\n";

    if (opcode == "tassign") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tassign expects 2 operands");
      return "  TASSIGN(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ");\n";
    }

    if (opcode == "tmov") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tmov expects 2 operands");
      return assignPrefix() + "TMOV(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ");\n";
    }

    if (opcode == "tadd") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("tadd expects 3 operands");
      return assignPrefix() + "TADD(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " +
             resolve(operands[2]) + ");\n";
    }

    if (opcode == "tmatmul") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("tmatmul expects 3 operands");
      return assignPrefix() + "TMATMUL(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " +
             resolve(operands[2]) + ");\n";
    }

    if (opcode == "tload") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tload expects 2 operands");
      auto dst = resolve(operands[0]);
      auto [base, idx] = parseIndexedOperand(operands[1]);
      auto src = resolve(base);
      if (!idx)
        return assignPrefix() + "TLOAD(" + dst + ", " + src + ");\n";
      auto r0 = resolve(idx->first);
      auto c0 = resolve(idx->second);
      if ((r0 == "0" || r0 == "c_r0") && (c0 == "0" || c0 == "c_c0"))
        return assignPrefix() + "TLOAD(" + dst + ", " + src + ");\n";

      std::ostringstream ss;
      ss << "  // NOTE: tload with non-zero indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << src << "_ptr = " << src << ".data();\n";
      ss << "  decltype(" << src << ") " << src << "_view(" << src << "_ptr);\n";
      ss << "  auto " << src << "_off = (" << r0 << ") * " << src
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << src
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << src << "_view, " << src << "_ptr + " << src << "_off);\n";
      ss << assignPrefix() << "TLOAD(" << dst << ", " << src << "_view);\n";
      return ss.str();
    }

    if (opcode == "tstore") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tstore expects 2 operands");
      auto [base, idx] = parseIndexedOperand(operands[0]);
      auto dst = resolve(base);
      auto src = resolve(operands[1]);
      if (!idx)
        return assignPrefix() + "TSTORE(" + dst + ", " + src + ");\n";
      auto r0 = resolve(idx->first);
      auto c0 = resolve(idx->second);
      if ((r0 == "0" || r0 == "c_r0") && (c0 == "0" || c0 == "c_c0"))
        return assignPrefix() + "TSTORE(" + dst + ", " + src + ");\n";

      std::ostringstream ss;
      ss << "  // NOTE: tstore with non-zero indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << dst << "_ptr = " << dst << ".data();\n";
      ss << "  decltype(" << dst << ") " << dst << "_view(" << dst << "_ptr);\n";
      ss << "  auto " << dst << "_off = (" << r0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
      ss << assignPrefix() << "TSTORE(" << dst << "_view, " << src << ");\n";
      return ss.str();
    }

    if (opcode == "tsync") {
      std::vector<std::string> evs;
      if (auto arr = op->getAttrOfType<mlir::ArrayAttr>("events")) {
        for (auto a : arr) {
          auto s = llvm::dyn_cast<mlir::StringAttr>(a);
          if (!s)
            llvm::report_fatal_error("tsync events must be strings");
          evs.push_back(s.getValue().str());
        }
      } else {
        // Also accept `tsync %e0, %e1 : ...` directly from PTO-AS input, which is parsed
        // as an `operands = ["%e0", "%e1"]` attribute by the frontend.
        auto operands = readOperands(op);
        for (auto &o : operands)
          evs.push_back(resolve(o));
      }
      std::ostringstream ss;
      ss << "  TSYNC(";
      for (size_t i = 0; i < evs.size(); ++i) {
        if (i)
          ss << ", ";
        ss << evs[i];
      }
      ss << ");\n";
      return ss.str();
    }

    // meta ops
    if (opcode == "record_event")
      return "";

    llvm::report_fatal_error(llvm::Twine("Unsupported opcode for CCE emission: ") + opcode);
  };

  auto emitBlock = [&](mlir::Block &block, int depth, auto &&self) -> void {
    std::string extra(depth * 2, ' ');

    for (auto it = block.begin(); it != block.end(); ++it) {
      auto *op = &*it;
      auto name = op->getName().getStringRef();
      if (name == "pto.arg" || name == "pto.const" || name == "pto.record_event")
        continue;

      // `record_event` assignment is a lookahead within the same block.
      const std::string *assignEvent = nullptr;
      std::string assignBuf;
      auto next = std::next(it);
      if (next != block.end() && next->getName().getStringRef() == "pto.record_event") {
        auto e = readRecordEvent(&*next);
        assignBuf = e.name;
        assignEvent = &assignBuf;
      }

      if (name == "scf.for") {
        auto operands = readOperands(op);
        if (operands.size() != 4)
          llvm::report_fatal_error("scf.for expects 4 operands (%iv, lb, ub, step)");
        auto iv = trim(operands[0]);
        if (!iv.empty() && iv[0] == '%')
          iv = iv.substr(1);
        auto lb = resolve(operands[1]);
        auto ub = resolve(operands[2]);
        auto step = resolve(operands[3]);
        os << std::string(2 + depth * 2, ' ') << "for (int64_t " << iv << " = " << lb << "; " << iv << " < " << ub
           << "; " << iv << " += " << step << ") {\n";
        if (!op->getRegions().empty() && !op->getRegion(0).empty())
          self(op->getRegion(0).front(), depth + 1, self);
        os << std::string(2 + depth * 2, ' ') << "}\n";
        continue;
      }

      if (name == "scf.if") {
        auto operands = readOperands(op);
        if (operands.size() != 1)
          llvm::report_fatal_error("scf.if expects 1 operand (%cond)");
        auto cond = resolve(operands[0]);
        os << std::string(2 + depth * 2, ' ') << "if (" << cond << ") {\n";
        if (op->getNumRegions() >= 1 && !op->getRegion(0).empty())
          self(op->getRegion(0).front(), depth + 1, self);
        bool hasElse = false;
        if (op->getNumRegions() >= 2 && !op->getRegion(1).empty()) {
          // Consider else present if the else block contains any non-yield ops.
          for (auto &op2 : op->getRegion(1).front().getOperations()) {
            if (op2.getName().getStringRef() != "scf.yield") {
              hasElse = true;
              break;
            }
          }
        }
        if (hasElse) {
          os << std::string(2 + depth * 2, ' ') << "} else {\n";
          self(op->getRegion(1).front(), depth + 1, self);
          os << std::string(2 + depth * 2, ' ') << "}\n";
        } else {
          os << std::string(2 + depth * 2, ' ') << "}\n";
        }
        continue;
      }

      // Default: PTO instruction in the current block.
      os << indentExtra(emitInstrCall(op, assignEvent), extra);
    }
  };

  emitBlock(*module.getBody(), 0, emitBlock);

  os << "}\n";
  return os.str();
}

std::string emitCpuCppFromModule(mlir::ModuleOp module, const std::string &repoRoot) {
  std::vector<ArgInfo> args;
  std::vector<ConstInfo> consts;

  for (auto &op : module.getBody()->getOperations()) {
    auto name = op.getName().getStringRef();
    if (name == "pto.arg") {
      auto n = op.getAttrOfType<mlir::StringAttr>("name");
      auto t = op.getAttrOfType<mlir::StringAttr>("type");
      if (!n || !t)
        llvm::report_fatal_error("pto.arg missing attrs");
      args.push_back({n.getValue().str(), t.getValue().str()});
      continue;
    }
    if (name == "pto.const") {
      auto n = op.getAttrOfType<mlir::StringAttr>("name");
      auto v = op.getAttrOfType<mlir::StringAttr>("value");
      auto t = op.getAttrOfType<mlir::StringAttr>("type");
      if (!n || !v || !t)
        llvm::report_fatal_error("pto.const missing attrs");
      consts.push_back({n.getValue().str(), v.getValue().str(), t.getValue().str()});
      continue;
    }
  }

  // Classify args.
  struct TensorArg {
    ArgInfo a;
    std::map<std::string, std::string> kv;
  };
  struct TileLocal {
    ArgInfo a;
    std::map<std::string, std::string> kv;
  };

  std::vector<TensorArg> tensorArgs;
  std::vector<TileLocal> tileLocals;
  for (auto &a : args) {
    if (isTensorType(a.typeStr)) {
      auto kvOpt = tryParseAngleKVs(a.typeStr, "!pto.tensor");
      if (!kvOpt)
        kvOpt = tryParseAngleKVs(a.typeStr, "!pto.gtensor"); // compat
      if (!kvOpt)
        llvm::report_fatal_error(llvm::Twine("tensor type parse failed: ") + a.typeStr);
      auto kv = *kvOpt;
      if (kv.find("dtype") == kv.end() && kv.find("element") != kv.end())
        kv["dtype"] = kv["element"];
      tensorArgs.push_back({a, std::move(kv)});
    } else if (isTileType(a.typeStr)) {
      auto kvOpt = tryParseAngleKVs(a.typeStr, "!pto.tile");
      if (!kvOpt)
        llvm::report_fatal_error(llvm::Twine("tile type parse failed: ") + a.typeStr);
      auto kv = *kvOpt;
      if (kv.find("dtype") == kv.end() && kv.find("element") != kv.end())
        kv["dtype"] = kv["element"];
      tileLocals.push_back({a, std::move(kv)});
    }
  }

  std::ostringstream os;
  os << "// Generated by ptoas (CPU simulator)\n";
  os << "#define __CPU_SIM\n";
  os << "#include <pto/pto-inst.hpp>\n";
  os << "#include <cstdint>\n";
  os << "using namespace pto;\n\n";

  os << "extern \"C\" void pto_kernel_cpu(";
  for (size_t i = 0; i < tensorArgs.size(); ++i) {
    if (i)
      os << ", ";
    os << "void* " << tensorArgs[i].a.name.substr(1);
  }
  os << ") {\n";

  // Tensor objects.
  for (auto &t : tensorArgs) {
    auto &kv = t.kv;
    auto dtype = kv.at("dtype");
    auto shape = parseList5(kv.at("shape"));
    auto stride = kv.count("stride") ? parseList5(kv.at("stride")) : defaultStrideForShape5(shape);
    auto layout = kv.count("layout") ? kv.at("layout") : "ND";
    auto elemCpp = elemToCpp(dtype);
    auto baseName = t.a.name.substr(1);
    os << "  auto* " << baseName << "_ptr = (" << elemCpp << "*)" << baseName << ";\n";
    os << "  using " << baseName << "_Shape = Shape<" << intOrDynamic(shape[0]) << ", " << intOrDynamic(shape[1])
       << ", " << intOrDynamic(shape[2]) << ", " << intOrDynamic(shape[3]) << ", " << intOrDynamic(shape[4]) << ">;\n";
    os << "  using " << baseName << "_Stride = Stride<" << intOrDynamic(stride[0]) << ", " << intOrDynamic(stride[1])
       << ", " << intOrDynamic(stride[2]) << ", " << intOrDynamic(stride[3]) << ", " << intOrDynamic(stride[4])
       << ">;\n";
    os << "  using " << baseName << "_Tensor = GlobalTensor<" << elemCpp << ", " << baseName << "_Shape, " << baseName
       << "_Stride, " << cppLayout(layout) << ">;\n";
    os << "  " << baseName << "_Tensor g_" << baseName << "(" << baseName << "_ptr);\n\n";
  }

  // Tile locals.
  for (auto &t : tileLocals) {
    auto &kv = t.kv;
    auto dtype = kv.at("dtype");
    auto elemCpp = elemToCpp(dtype);
    auto rows = intOrDynamic(kv.at("rows"));
    auto cols = intOrDynamic(kv.at("cols"));
    auto loc = kv.count("loc") ? kv.at("loc") : "Vec";
    auto blayout = kv.count("blayout") ? kv.at("blayout") : "RowMajor";
    auto slayout = kv.count("slayout") ? kv.at("slayout") : "NoneBox";
    auto fractal = kv.count("fractal") ? kv.at("fractal") : defaultFractalBytesForTileLoc(loc);
    auto pad = kv.count("pad") ? kv.at("pad") : "Null";
    auto valid = kv.count("valid") ? kv.at("valid") : (kv.at("rows") + "x" + kv.at("cols"));
    auto x = valid.find('x');
    if (x == std::string::npos)
      llvm::report_fatal_error("tile valid must be RxC");
    auto vrow = intOrDynamic(valid.substr(0, x));
    auto vcol = intOrDynamic(valid.substr(x + 1));

    auto baseName = t.a.name.substr(1);
    os << "  using " << baseName << "_Tile = Tile<" << cppTileType(loc) << ", " << elemCpp << ", " << rows << ", "
       << cols << ", " << cppBLayout(blayout) << ", " << vrow << ", " << vcol << ", " << cppSLayout(slayout) << ", "
       << intOrDynamic(fractal) << ", " << cppPad(pad) << ">;\n";
    os << "  " << baseName << "_Tile t_" << baseName << ";\n";
  }
  if (!tileLocals.empty())
    os << "\n";

  // Constants.
  for (auto &c : consts) {
    auto name = c.name.substr(1);
    auto v = trim(c.value);
    if (v.rfind("0x", 0) == 0 || v.rfind("0X", 0) == 0)
      os << "  constexpr uint64_t c_" << name << " = " << v << ";\n";
    else
      os << "  constexpr int64_t c_" << name << " = " << v << ";\n";
  }
  if (!consts.empty())
    os << "\n";

  auto resolve = [&](const std::string &v) -> std::string {
    auto t = trim(v);
    if (!t.empty() && t[0] == '%') {
      auto key = t.substr(1);
      for (auto &ta : tensorArgs)
        if (ta.a.name.substr(1) == key)
          return "g_" + key;
      for (auto &tl : tileLocals)
        if (tl.a.name.substr(1) == key)
          return "t_" + key;
      for (auto &c : consts)
        if (c.name.substr(1) == key)
          return "c_" + key;
      return key;
    }
    return t;
  };

  auto emitInstrCallCpu = [&](mlir::Operation *op) -> std::string {
    auto opcode = mnemonicFor(op->getName().getStringRef());

    if (opcode == "prologue")
      return "  // prologue\n";
    if (opcode == "epilogue")
      return "  // epilogue\n";

    if (opcode == "get_block_idx") {
      auto operands = readOperands(op);
      if (operands.size() != 1)
        llvm::report_fatal_error("get_block_idx expects 1 operand (dest)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  int64_t " + dst + " = 0;\n";
    }
    if (opcode == "get_block_num") {
      auto operands = readOperands(op);
      if (operands.size() != 1)
        llvm::report_fatal_error("get_block_num expects 1 operand (dest)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  int64_t " + dst + " = 1;\n";
    }

    if (opcode == "iadd") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("iadd expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") + (" + resolve(operands[2]) + ");\n";
    }
    if (opcode == "imul") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("imul expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") * (" + resolve(operands[2]) + ");\n";
    }

    auto emitIcmp = [&](const char *expect, const char *cxxOp) -> std::optional<std::string> {
      if (opcode != expect)
        return std::nullopt;
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error(llvm::Twine(expect) + " expects 3 operands (dst, lhs, rhs)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") " + cxxOp + " (" + resolve(operands[2]) + ");\n";
    };

    if (auto s = emitIcmp("icmp_eq", "=="))
      return *s;
    if (auto s = emitIcmp("icmp_ne", "!="))
      return *s;
    if (auto s = emitIcmp("icmp_lt", "<"))
      return *s;
    if (auto s = emitIcmp("icmp_le", "<="))
      return *s;
    if (auto s = emitIcmp("icmp_gt", ">"))
      return *s;
    if (auto s = emitIcmp("icmp_ge", ">="))
      return *s;

    if (opcode == "scf.yield")
      return "  // scf.yield\n";

    if (opcode == "tassign") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tassign expects 2 operands");
      return "  TASSIGN(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ");\n";
    }
    if (opcode == "tmov") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tmov expects 2 operands");
      return "  TMOV(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ");\n";
    }
    if (opcode == "tadd") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("tadd expects 3 operands");
      return "  TADD(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " + resolve(operands[2]) + ");\n";
    }
    if (opcode == "tmatmul") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("tmatmul expects 3 operands");
      return "  TMATMUL(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " + resolve(operands[2]) + ");\n";
    }

    if (opcode == "tload") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tload expects 2 operands");
      auto dst = resolve(operands[0]);
      auto [base, idx] = parseIndexedOperand(operands[1]);
      auto src = resolve(base);
      if (!idx)
        return "  TLOAD(" + dst + ", " + src + ");\n";
      auto r0 = resolve(idx->first);
      auto c0 = resolve(idx->second);
      if ((r0 == "0" || r0 == "c_r0") && (c0 == "0" || c0 == "c_c0"))
        return "  TLOAD(" + dst + ", " + src + ");\n";

      std::ostringstream ss;
      ss << "  // NOTE: tload with non-zero indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << src << "_ptr = " << src << ".data();\n";
      ss << "  decltype(" << src << ") " << src << "_view(" << src << "_ptr);\n";
      ss << "  auto " << src << "_off = (" << r0 << ") * " << src
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << src
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << src << "_view, " << src << "_ptr + " << src << "_off);\n";
      ss << "  TLOAD(" << dst << ", " << src << "_view);\n";
      return ss.str();
    }

    if (opcode == "tstore") {
      auto operands = readOperands(op);
      if (operands.size() != 2)
        llvm::report_fatal_error("tstore expects 2 operands");
      auto [base, idx] = parseIndexedOperand(operands[0]);
      auto dst = resolve(base);
      auto src = resolve(operands[1]);
      if (!idx)
        return "  TSTORE(" + dst + ", " + src + ");\n";
      auto r0 = resolve(idx->first);
      auto c0 = resolve(idx->second);
      if ((r0 == "0" || r0 == "c_r0") && (c0 == "0" || c0 == "c_c0"))
        return "  TSTORE(" + dst + ", " + src + ");\n";

      std::ostringstream ss;
      ss << "  // NOTE: tstore with non-zero indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << dst << "_ptr = " << dst << ".data();\n";
      ss << "  decltype(" << dst << ") " << dst << "_view(" << dst << "_ptr);\n";
      ss << "  auto " << dst << "_off = (" << r0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
      ss << "  TSTORE(" << dst << "_view, " << src << ");\n";
      return ss.str();
    }

    llvm::report_fatal_error(llvm::Twine("Unsupported opcode for CPU emission: ") + opcode);
  };

  auto emitBlock = [&](mlir::Block &block, int depth, auto &&self) -> void {
    std::string extra(depth * 2, ' ');

    for (auto &op : block.getOperations()) {
      auto name = op.getName().getStringRef();
      if (name == "pto.arg" || name == "pto.const" || name == "pto.record_event" || name == "pto.tsync")
        continue;

      if (name == "scf.for") {
        auto operands = readOperands(&op);
        if (operands.size() != 4)
          llvm::report_fatal_error("scf.for expects 4 operands (%iv, lb, ub, step)");
        auto iv = trim(operands[0]);
        if (!iv.empty() && iv[0] == '%')
          iv = iv.substr(1);
        auto lb = resolve(operands[1]);
        auto ub = resolve(operands[2]);
        auto step = resolve(operands[3]);
        os << std::string(2 + depth * 2, ' ') << "for (int64_t " << iv << " = " << lb << "; " << iv << " < " << ub
           << "; " << iv << " += " << step << ") {\n";
        if (!op.getRegions().empty() && !op.getRegion(0).empty())
          self(op.getRegion(0).front(), depth + 1, self);
        os << std::string(2 + depth * 2, ' ') << "}\n";
        continue;
      }

      if (name == "scf.if") {
        auto operands = readOperands(&op);
        if (operands.size() != 1)
          llvm::report_fatal_error("scf.if expects 1 operand (%cond)");
        auto cond = resolve(operands[0]);
        os << std::string(2 + depth * 2, ' ') << "if (" << cond << ") {\n";
        if (op.getNumRegions() >= 1 && !op.getRegion(0).empty())
          self(op.getRegion(0).front(), depth + 1, self);
        bool hasElse = false;
        if (op.getNumRegions() >= 2 && !op.getRegion(1).empty()) {
          for (auto &op2 : op.getRegion(1).front().getOperations()) {
            if (op2.getName().getStringRef() != "scf.yield") {
              hasElse = true;
              break;
            }
          }
        }
        if (hasElse) {
          os << std::string(2 + depth * 2, ' ') << "} else {\n";
          self(op.getRegion(1).front(), depth + 1, self);
          os << std::string(2 + depth * 2, ' ') << "}\n";
        } else {
          os << std::string(2 + depth * 2, ' ') << "}\n";
        }
        continue;
      }

      os << indentExtra(emitInstrCallCpu(&op), extra);
    }
  };

  emitBlock(*module.getBody(), 0, emitBlock);

  os << "}\n";
  return os.str();
}

} // namespace ptoas
