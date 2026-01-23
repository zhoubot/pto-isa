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

static std::vector<std::string> parseList2or5(const std::string &s) {
  auto t = trim(s);
  if (t.size() < 2 || t.front() != '[' || t.back() != ']')
    llvm::report_fatal_error(llvm::Twine("Expected list literal [..]: ") + t);
  std::vector<std::string> items;
  std::string inner = t.substr(1, t.size() - 2);
  for (auto &p : splitTopLevelCommas(inner))
    items.push_back(trim(p));
  if (items.size() != 2 && items.size() != 5)
    llvm::report_fatal_error(llvm::Twine("Expected 2 or 5 elements: ") + t);
  return items;
}

static std::vector<std::string> shapeTo5(const std::vector<std::string> &shape) {
  if (shape.size() == 5)
    return shape;
  if (shape.size() == 2)
    return {"1", "1", "1", shape[0], shape[1]};
  llvm::report_fatal_error("shape must be 2-D or 5-D");
}

static std::vector<std::string> strideTo5(const std::vector<std::string> &stride) {
  if (stride.size() == 5)
    return stride;
  if (stride.size() == 2)
    return {"1", "1", "1", stride[0], stride[1]};
  llvm::report_fatal_error("stride must be 2-D or 5-D");
}

static std::vector<std::string> offsetTo5(const std::vector<std::string> &offset) {
  if (offset.size() == 5)
    return offset;
  if (offset.size() == 2)
    return {"0", "0", "0", offset[0], offset[1]};
  llvm::report_fatal_error("offsets must be 2-D or 5-D");
}

static std::string extractBracketListOrEmpty(const std::string &s, llvm::StringRef key) {
  auto pos = s.find(key.str());
  if (pos == std::string::npos)
    return "";
  pos += key.size();
  while (pos < s.size() && std::isspace((unsigned char)s[pos]))
    pos++;
  if (pos >= s.size() || s[pos] != '[')
    return "";
  int depth = 0;
  for (size_t i = pos; i < s.size(); ++i) {
    if (s[i] == '[')
      depth++;
    else if (s[i] == ']') {
      depth--;
      if (depth == 0)
        return s.substr(pos, i - pos + 1);
    }
  }
  return "";
}

static std::string extractScalarAfterKeyOrEmpty(const std::string &s, llvm::StringRef key) {
  auto pos = s.find(key.str());
  if (pos == std::string::npos)
    return "";
  pos += key.size();
  while (pos < s.size() && std::isspace((unsigned char)s[pos]))
    pos++;
  size_t end = pos;
  while (end < s.size()) {
    char c = s[end];
    if (std::isspace((unsigned char)c) || c == ',' || c == '}' || c == ':')
      break;
    end++;
  }
  return trim(s.substr(pos, end - pos));
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

static bool isTileType(const std::string &typeStr) { return llvm::StringRef(typeStr).starts_with("!pto.tile<"); }

static bool isTensorType(const std::string &typeStr) {
  auto s = llvm::StringRef(typeStr);
  return s.starts_with("!pto.tensor<") || s.starts_with("!pto.gtensor<");
}

static std::vector<std::string> readOperands(mlir::Operation *op);

struct MakeTensorViewInfo {
  std::string viewName; // with leading %
  std::string baseArg;  // with leading % (e.g. %arg0)
  std::string typeStr;  // !pto.tensor<...>
};

struct AllocTileInfo {
  std::string tileName;                 // with leading %
  std::string typeStr;                  // !pto.tile<...>
  std::optional<std::string> addrValue; // optional address operand (e.g. %addr_x)
};

static std::string buildTensorTypeFromMakeView(mlir::Operation *op) {
  auto operands = readOperands(op);
  if (operands.size() < 2)
    llvm::report_fatal_error("pto.make_tensor_view expects at least 2 operands (%view, %argN)");

  auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig");
  if (typeSig && llvm::StringRef(typeSig.getValue()).starts_with("!pto.tensor"))
    return typeSig.getValue().str();

  // Merge leftover tokens; the frontend splits by top-level commas only, so multiple `k=v`
  // pairs may end up in a single operand when the input omits commas between them.
  std::string opts;
  for (size_t i = 2; i < operands.size(); ++i) {
    if (!opts.empty())
      opts += ", ";
    opts += operands[i];
  }

  auto dtype = extractScalarAfterKeyOrEmpty(opts, "dtype=");
  if (dtype.empty())
    dtype = extractScalarAfterKeyOrEmpty(opts, "element=");
  if (dtype.empty())
    llvm::report_fatal_error("pto.make_tensor_view missing dtype=...");

  auto layout = extractScalarAfterKeyOrEmpty(opts, "layout=");
  if (layout.empty())
    layout = "ND";

  auto shapeLit = extractBracketListOrEmpty(opts, "shape=");
  if (shapeLit.empty())
    llvm::report_fatal_error("pto.make_tensor_view missing shape=[...]");

  auto strideLit = extractBracketListOrEmpty(opts, "strides=");
  if (strideLit.empty())
    strideLit = extractBracketListOrEmpty(opts, "stride=");

  auto shape = parseList2or5(shapeLit);
  std::vector<std::string> stride;
  if (!strideLit.empty()) {
    stride = parseList2or5(strideLit);
  } else {
    if (shape.size() == 2)
      stride = {shape[1], "1"};
    else
      stride = defaultStrideForShape5(shapeTo5(shape));
  }

  std::ostringstream ss;
  ss << "!pto.tensor<dtype=" << dtype << ", shape=[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      ss << ",";
    ss << shape[i];
  }
  ss << "], stride=[";
  for (size_t i = 0; i < stride.size(); ++i) {
    if (i)
      ss << ",";
    ss << stride[i];
  }
  ss << "], layout=" << layout << ">";
  return ss.str();
}

struct SubviewInfo {
  std::string viewName;                 // with leading %
  std::string baseView;                 // with leading % (view or %argN)
  std::vector<std::string> offsets5;    // 5D offsets (DIM_0..DIM_4)
  std::optional<std::string> typeStr;   // optional explicit typesig (must match base)
};

static SubviewInfo readSubview(mlir::Operation *op) {
  auto operands = readOperands(op);
  if (operands.size() < 2)
    llvm::report_fatal_error("pto.subview expects at least 2 operands (%view, %base)");

  std::string opts;
  for (size_t i = 2; i < operands.size(); ++i) {
    if (!opts.empty())
      opts += ", ";
    opts += operands[i];
  }

  auto offsetsLit = extractBracketListOrEmpty(opts, "offsets=");
  if (offsetsLit.empty())
    offsetsLit = extractBracketListOrEmpty(opts, "offset=");
  if (offsetsLit.empty())
    llvm::report_fatal_error("pto.subview missing offsets=[...]");

  SubviewInfo out;
  out.viewName = trim(operands[0]);
  out.baseView = trim(operands[1]);
  out.offsets5 = offsetTo5(parseList2or5(offsetsLit));
  if (auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig"))
    out.typeStr = typeSig.getValue().str();
  return out;
}

static AllocTileInfo readAllocTile(mlir::Operation *op) {
  auto operands = readOperands(op);
  if (operands.empty())
    llvm::report_fatal_error("pto.alloc_tile expects at least 1 operand (%tile)");
  auto typeSig = op->getAttrOfType<mlir::StringAttr>("typesig");
  if (!typeSig || !llvm::StringRef(typeSig.getValue()).starts_with("!pto.tile"))
    llvm::report_fatal_error("pto.alloc_tile missing ': !pto.tile<...>' type");
  AllocTileInfo out;
  out.tileName = trim(operands[0]);
  out.typeStr = typeSig.getValue().str();
  if (operands.size() >= 2)
    out.addrValue = trim(operands[1]);
  return out;
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

std::string emitCceFromModule(mlir::ModuleOp module, const std::string &repoRoot, const std::string &memoryModel,
                              const std::string &kernelName) {
  std::vector<ArgInfo> args;
  std::vector<ConstInfo> consts;
  std::vector<MakeTensorViewInfo> makeViews;
  std::vector<AllocTileInfo> allocTiles;

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
    if (name == "pto.make_tensor_view") {
      auto operands = readOperands(&op);
      if (operands.size() < 2)
        llvm::report_fatal_error("pto.make_tensor_view expects: %view, %argN, ...");
      makeViews.push_back({trim(operands[0]), trim(operands[1]), buildTensorTypeFromMakeView(&op)});
      continue;
    }
    if (name == "pto.alloc_tile") {
      allocTiles.push_back(readAllocTile(&op));
      continue;
    }
  }

  // Const map (for substituting %c* inside make_tensor_view type spellings).
  std::map<std::string, std::string> constMap;
  for (auto &c : consts)
    constMap[c.name] = trim(c.value);

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

  // New-format declarations:
  // - `pto.make_tensor_view` introduces a view symbol that aliases a kernel base argument.
  // - `pto.alloc_tile` introduces a local tile value (optionally pre-bound to an address).
  std::map<std::string, std::string> tensorViewAlias; // "%view" -> "%argN"

  auto ensureArg = [&](const std::string &name, const std::string &typeStr) {
    for (auto &a : args) {
      if (a.name == name) {
        if (a.typeStr != typeStr)
          llvm::report_fatal_error("conflicting types for the same symbol");
        return;
      }
    }
    args.push_back({name, typeStr});
  };

  for (auto &mv : makeViews) {
    tensorViewAlias[mv.viewName] = mv.baseArg;
    ensureArg(mv.baseArg, mv.typeStr);
  }
  for (auto &at : allocTiles)
    ensureArg(at.tileName, at.typeStr);

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
  // Keep the emitted source buildable as a regular `.cpp` translation unit on non-CCE toolchains.
  // The real kernel body is only compiled by the Ascend CCE compiler (`bisheng -xcce` / `ccec`),
  // which defines `__CCE_AICORE__` (see demos/baseline/* kernels).
  os << "#if defined(__CCE_AICORE__)\n";
  os << "#define " << memoryModel << "\n";
  os << "#include \"kernel_operator.h\"\n";
  os << "#include <pto/pto-inst.hpp>\n";
  os << "#include <cstdint>\n";
  os << "using namespace pto;\n\n";

  // Kernel signature: tensors only.
  os << "extern \"C\" __global__ AICORE void " << kernelName << "(";
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
    auto shape = shapeTo5(parseList2or5(kv.at("shape")));
    for (auto &v : shape)
      if (!v.empty() && v[0] == '%')
        v = constMap.at(v);
    std::vector<std::string> stride;
    if (kv.count("stride")) {
      stride = strideTo5(parseList2or5(kv.at("stride")));
      for (auto &v : stride)
        if (!v.empty() && v[0] == '%')
          v = constMap.at(v);
    } else {
      stride = defaultStrideForShape5(shape);
    }
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

  // Local tensor variables introduced by meta ops (e.g. `pto.subview`).
  std::map<std::string, std::string> localTensorVars; // "%view" -> "g_view"

  struct TileDims {
    std::string rows;
    std::string cols;
  };
  struct TensorInfo {
    std::string elemCpp;
    std::string strideTypeName;
    std::string layoutCpp;
  };

  // Resolved symbol -> tile shape (rows/cols).
  std::map<std::string, TileDims> tileDimsByResolvedName;
  for (auto &tl : tileLocals) {
    auto baseName = tl.a.name.substr(1);
    auto rows = tl.kv.at("rows");
    auto cols = tl.kv.at("cols");
    if (!rows.empty() && rows[0] == '%')
      rows = constMap.at(rows);
    if (!cols.empty() && cols[0] == '%')
      cols = constMap.at(cols);
    tileDimsByResolvedName["t_" + baseName] = {intOrDynamic(rows), intOrDynamic(cols)};
  }

  // Resolved symbol -> tensor ABI info (elem/stride/layout).
  std::map<std::string, TensorInfo> tensorInfoByResolvedName;
  for (auto &ta : tensorArgs) {
    auto &kv = ta.kv;
    auto dtype = kv.at("dtype");
    auto elemCpp = elemToCpp(dtype);
    auto layout = kv.count("layout") ? kv.at("layout") : "ND";
    auto baseName = ta.a.name.substr(1);
    tensorInfoByResolvedName["g_" + baseName] = {elemCpp, baseName + "_Stride", cppLayout(layout)};
  }

  auto resolve = [&](const std::string &v) -> std::string {
    auto t = trim(v);
    if (!t.empty() && t[0] == '%') {
      auto aliasIt = tensorViewAlias.find(t);
      if (aliasIt != tensorViewAlias.end())
        t = aliasIt->second;
      auto localIt = localTensorVars.find(t);
      if (localIt != localTensorVars.end())
        return localIt->second;
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

  auto emitSubviewStmt = [&](mlir::Operation *op) -> std::string {
    auto sv = readSubview(op);
    if (sv.viewName.empty() || sv.viewName[0] != '%')
      llvm::report_fatal_error("pto.subview destination must be a %name symbol");

    // Resolve base before registering the new view to avoid accidental self-reference.
    auto baseVar = resolve(sv.baseView);
    auto viewKey = sv.viewName.substr(1);
    auto viewVar = "g_" + viewKey;
    localTensorVars[sv.viewName] = viewVar;
    if (auto it = tensorInfoByResolvedName.find(baseVar); it != tensorInfoByResolvedName.end())
      tensorInfoByResolvedName[viewVar] = it->second;

    std::ostringstream ss;
    ss << "  auto* " << viewVar << "_base = " << baseVar << ".data();\n";
    ss << "  decltype(" << baseVar << ") " << viewVar << "(" << viewVar << "_base);\n";
    ss << "  auto " << viewVar << "_off = (" << resolve(sv.offsets5[0]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_0) + (" << resolve(sv.offsets5[1]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_1) + (" << resolve(sv.offsets5[2]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_2) + (" << resolve(sv.offsets5[3]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_3) + (" << resolve(sv.offsets5[4]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_4);\n";
    ss << "  TASSIGN(" << viewVar << ", " << viewVar << "_base + " << viewVar << "_off);\n";
    return ss.str();
  };

  // Tile address binding (new-format PTO-AS removes explicit `tassign`).
  for (auto &at : allocTiles) {
    if (!at.addrValue)
      continue;
    os << "  TASSIGN(" << resolve(at.tileName) << ", " << resolve(*at.addrValue) << ");\n";
  }
  if (!allocTiles.empty())
    os << "\n";

  auto emitInstrCall = [&](mlir::Operation *op, const std::string *assignEvent) -> std::string {
    auto opcode = mnemonicFor(op->getName().getStringRef());
    (void)assignEvent;

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

    if (opcode == "idiv") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("idiv expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") / (" + resolve(operands[2]) + ");\n";
    }

    if (opcode == "irem") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("irem expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") % (" + resolve(operands[2]) + ");\n";
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

    if (opcode == "tmatmul_acc") {
      auto operands = readOperands(op);
      if (operands.size() != 4)
        llvm::report_fatal_error("tmatmul_acc expects 4 operands");
      return "  TMATMUL_ACC(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " + resolve(operands[2]) +
             ", " + resolve(operands[3]) + ");\n";
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

      auto ti = tensorInfoByResolvedName.find(src);
      auto td = tileDimsByResolvedName.find(dst);
      if (ti != tensorInfoByResolvedName.end() && td != tileDimsByResolvedName.end()) {
        std::ostringstream ss;
        ss << "  // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.\n";
        ss << "  {\n";
        ss << "    auto* " << src << "_ptr = " << src << ".data();\n";
        ss << "    auto " << src << "_off = (" << r0 << ") * " << src
           << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << src
           << ".GetStride(GlobalTensorDim::DIM_4);\n";
        ss << "    using TloadShape = Shape<1, 1, 1, " << td->second.rows << ", " << td->second.cols << ">;\n";
        ss << "    using TloadTensor = GlobalTensor<" << ti->second.elemCpp << ", TloadShape, "
           << ti->second.strideTypeName << ", " << ti->second.layoutCpp << ">;\n";
        ss << "    TloadTensor " << src << "_view(" << src << "_ptr);\n";
        ss << "    TASSIGN(" << src << "_view, " << src << "_ptr + " << src << "_off);\n";
        ss << "    TLOAD(" << dst << ", " << src << "_view);\n";
        ss << "  }\n";
        return ss.str();
      }

      std::ostringstream ss;
      ss << "  // NOTE: tload with indices is lowered via pointer bump (prototype).\n";
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

      auto ti = tensorInfoByResolvedName.find(dst);
      auto td = tileDimsByResolvedName.find(src);
      if (ti != tensorInfoByResolvedName.end() && td != tileDimsByResolvedName.end()) {
        std::ostringstream ss;
        ss << "  // NOTE: tstore with indices uses a tile-shaped GlobalTensor view for conversion correctness.\n";
        ss << "  {\n";
        ss << "    auto* " << dst << "_ptr = " << dst << ".data();\n";
        ss << "    auto " << dst << "_off = (" << r0 << ") * " << dst
           << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
           << ".GetStride(GlobalTensorDim::DIM_4);\n";
        ss << "    using TstoreShape = Shape<1, 1, 1, " << td->second.rows << ", " << td->second.cols << ">;\n";
        ss << "    using TstoreTensor = GlobalTensor<" << ti->second.elemCpp << ", TstoreShape, "
           << ti->second.strideTypeName << ", " << ti->second.layoutCpp << ">;\n";
        ss << "    TstoreTensor " << dst << "_view(" << dst << "_ptr);\n";
        ss << "    TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
        ss << "    TSTORE(" << dst << "_view, " << src << ");\n";
        ss << "  }\n";
        return ss.str();
      }

      std::ostringstream ss;
      ss << "  // NOTE: tstore with indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << dst << "_ptr = " << dst << ".data();\n";
      ss << "  decltype(" << dst << ") " << dst << "_view(" << dst << "_ptr);\n";
      ss << "  auto " << dst << "_off = (" << r0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
      ss << "  TSTORE(" << dst << "_view, " << src << ");\n";
      return ss.str();
    }

    // Explicit event primitives (prototype): lower to set_flag/wait_flag.
    auto pipeForOpEnum = [&](llvm::StringRef opEnum) -> llvm::StringRef {
      if (opEnum == "TLOAD")
        return "MTE2";
      if (opEnum == "TSTORE_VEC" || opEnum == "TSTORE_MAT")
        return "MTE3";
      if (opEnum == "TSTORE_ACC")
        return "FIX";
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
      if (opEnum == "TCI")
        return "S";
      // Default: most remaining PTO ops are vector pipe on A2/A3 (see `include/pto/npu/a2a3/TSync.hpp`).
      if (opEnum.starts_with("T"))
        return "V";
      return "";
    };

    auto pipeConstForPipeName = [&](llvm::StringRef pipeName) -> std::string {
      if (pipeName == "S")
        return "PIPE_S";
      if (pipeName == "FIX")
        return "PIPE_FIX";
      if (pipeName == "V")
        return "PIPE_V";
      if (pipeName == "MTE1")
        return "PIPE_MTE1";
      if (pipeName == "MTE2")
        return "PIPE_MTE2";
      if (pipeName == "MTE3")
        return "PIPE_MTE3";
      if (pipeName == "M")
        return "PIPE_M";
      return "";
    };

    auto emitRecordOrWait = [&](llvm::StringRef kind) -> std::optional<std::string> {
      if (opcode != kind)
        return std::nullopt;

      auto normalizeOpValue = [&](std::string v) -> std::string {
        v = trim(std::move(v));
        // Accept `#pto.op<TLOAD>` or `#op<TLOAD>` spellings and extract the payload.
        if (!v.empty() && v[0] == '#') {
          auto l = v.find('<');
          auto r = v.rfind('>');
          if (l != std::string::npos && r != std::string::npos && r > l + 1)
            v = trim(v.substr(l + 1, r - l - 1));
        }
        return v;
      };

      std::optional<std::string> srcOpt;
      std::optional<std::string> dstOpt;
      std::optional<std::string> tokOpt;

      if (auto srcA = op->getAttrOfType<mlir::StringAttr>("src_op"))
        srcOpt = normalizeOpValue(srcA.getValue().str());
      if (auto dstA = op->getAttrOfType<mlir::StringAttr>("dst_op"))
        dstOpt = normalizeOpValue(dstA.getValue().str());
      if (auto tokA = op->getAttrOfType<mlir::StringAttr>("token"))
        tokOpt = trim(tokA.getValue().str());

      // Also accept textual `{...}` dict from PTO-AS input (stored as a single string attr `attrs` by the frontend).
      if ((!srcOpt || !dstOpt || !tokOpt) && op->hasAttr("attrs")) {
        auto raw = op->getAttrOfType<mlir::StringAttr>("attrs");
        if (raw) {
          auto s = trim(raw.getValue().str());
          if (!s.empty() && s.front() == '{' && s.back() == '}')
            s = trim(s.substr(1, s.size() - 2));
          for (auto &p : splitTopLevelCommas(s)) {
            auto eq = p.find('=');
            if (eq == std::string::npos)
              continue;
            auto k = trim(p.substr(0, eq));
            auto v = trim(p.substr(eq + 1));
            if (k == "src_op" && !srcOpt)
              srcOpt = normalizeOpValue(v);
            else if (k == "dst_op" && !dstOpt)
              dstOpt = normalizeOpValue(v);
            else if (k == "token" && !tokOpt)
              tokOpt = trim(v);
          }
        }
      }

      if (!srcOpt || !dstOpt || !tokOpt)
        llvm::report_fatal_error("pto.(record_event|wait_event) missing src_op/dst_op/token");

      auto src = *srcOpt;
      auto dst = *dstOpt;
      auto tok = *tokOpt;

      auto srcPipe = pipeForOpEnum(src);
      auto dstPipe = pipeForOpEnum(dst);
      if (srcPipe.empty() || dstPipe.empty())
        llvm::report_fatal_error("unknown pipe for src_op/dst_op");
      auto srcPipeConst = pipeConstForPipeName(srcPipe);
      auto dstPipeConst = pipeConstForPipeName(dstPipe);
      if (srcPipeConst.empty() || dstPipeConst.empty())
        llvm::report_fatal_error("unknown pipe constant for src/dst pipe");
      auto tokResolved = resolve(tok);
      std::ostringstream ss;
      ss << "  " << ((kind == "record_event") ? "set_flag" : "wait_flag") << "(" << srcPipeConst << ", "
         << dstPipeConst << ", static_cast<event_t>(" << tokResolved << "));\n";
      return ss.str();
    };

    if (auto s = emitRecordOrWait("record_event"))
      return *s;
    if (auto s = emitRecordOrWait("wait_event"))
      return *s;

    // Generic fallback: most PTO ISA ops have a corresponding C++ macro of the same name in uppercase.
    // Examples:
    //   trowmax -> TROWMAX
    //   tmuls   -> TMULS
    // This keeps the assembler toolchain usable while opcode-specific lowering evolves.
    auto operands = readOperands(op);
    if (operands.empty())
      return "  // " + opcode + "\n";
    std::string macro;
    macro.reserve(opcode.size());
    for (char ch : opcode)
      macro.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    std::ostringstream ss;
    ss << "  " << macro << "(";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i)
        ss << ", ";
      ss << resolve(operands[i]);
    }
    ss << ");\n";
    return ss.str();
  };

  auto emitBlock = [&](mlir::Block &block, int depth, auto &&self) -> void {
    std::string extra(depth * 2, ' ');

    for (auto it = block.begin(); it != block.end(); ++it) {
      auto *op = &*it;
      auto name = op->getName().getStringRef();
      if (name == "pto.subview") {
        os << indentExtra(emitSubviewStmt(op), extra);
        continue;
      }
      if (name == "pto.arg" || name == "pto.const" || name == "pto.make_tensor_view" || name == "pto.alloc_tile")
        continue;

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
      os << indentExtra(emitInstrCall(op, /*assignEvent=*/nullptr), extra);
    }
  };

  emitBlock(*module.getBody(), 0, emitBlock);

  os << "}\n";
  os << "#endif\n";
  return os.str();
}

std::string emitCpuCppFromModule(mlir::ModuleOp module, const std::string &repoRoot) {
  std::vector<ArgInfo> args;
  std::vector<ConstInfo> consts;
  std::vector<MakeTensorViewInfo> makeViews;
  std::vector<AllocTileInfo> allocTiles;

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
    if (name == "pto.make_tensor_view") {
      auto operands = readOperands(&op);
      if (operands.size() < 2)
        llvm::report_fatal_error("pto.make_tensor_view expects: %view, %argN, ...");
      makeViews.push_back({trim(operands[0]), trim(operands[1]), buildTensorTypeFromMakeView(&op)});
      continue;
    }
    if (name == "pto.alloc_tile") {
      allocTiles.push_back(readAllocTile(&op));
      continue;
    }
  }

  std::map<std::string, std::string> constMap;
  for (auto &c : consts)
    constMap[c.name] = trim(c.value);

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

  std::map<std::string, std::string> tensorViewAlias; // "%view" -> "%argN"
  auto ensureArg = [&](const std::string &name, const std::string &typeStr) {
    for (auto &a : args) {
      if (a.name == name) {
        if (a.typeStr != typeStr)
          llvm::report_fatal_error("conflicting types for the same symbol");
        return;
      }
    }
    args.push_back({name, typeStr});
  };

  for (auto &mv : makeViews) {
    tensorViewAlias[mv.viewName] = mv.baseArg;
    ensureArg(mv.baseArg, mv.typeStr);
  }
  for (auto &at : allocTiles)
    ensureArg(at.tileName, at.typeStr);

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
    auto shape = shapeTo5(parseList2or5(kv.at("shape")));
    for (auto &v : shape)
      if (!v.empty() && v[0] == '%')
        v = constMap.at(v);
    std::vector<std::string> stride;
    if (kv.count("stride")) {
      stride = strideTo5(parseList2or5(kv.at("stride")));
      for (auto &v : stride)
        if (!v.empty() && v[0] == '%')
          v = constMap.at(v);
    } else {
      stride = defaultStrideForShape5(shape);
    }
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

  std::map<std::string, std::string> localTensorVars; // "%view" -> "g_view"

  struct TileDims {
    std::string rows;
    std::string cols;
  };
  struct TensorInfo {
    std::string elemCpp;
    std::string strideTypeName;
    std::string layoutCpp;
  };

  // Resolved symbol -> tile shape (rows/cols).
  std::map<std::string, TileDims> tileDimsByResolvedName;
  for (auto &tl : tileLocals) {
    auto baseName = tl.a.name.substr(1);
    auto rows = tl.kv.at("rows");
    auto cols = tl.kv.at("cols");
    if (!rows.empty() && rows[0] == '%')
      rows = constMap.at(rows);
    if (!cols.empty() && cols[0] == '%')
      cols = constMap.at(cols);
    tileDimsByResolvedName["t_" + baseName] = {intOrDynamic(rows), intOrDynamic(cols)};
  }

  // Resolved symbol -> tensor ABI info (elem/stride/layout).
  std::map<std::string, TensorInfo> tensorInfoByResolvedName;
  for (auto &ta : tensorArgs) {
    auto &kv = ta.kv;
    auto dtype = kv.at("dtype");
    auto elemCpp = elemToCpp(dtype);
    auto layout = kv.count("layout") ? kv.at("layout") : "ND";
    auto baseName = ta.a.name.substr(1);
    tensorInfoByResolvedName["g_" + baseName] = {elemCpp, baseName + "_Stride", cppLayout(layout)};
  }

  auto resolve = [&](const std::string &v) -> std::string {
    auto t = trim(v);
    if (!t.empty() && t[0] == '%') {
      auto aliasIt = tensorViewAlias.find(t);
      if (aliasIt != tensorViewAlias.end())
        t = aliasIt->second;
      auto localIt = localTensorVars.find(t);
      if (localIt != localTensorVars.end())
        return localIt->second;
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

  auto emitSubviewStmt = [&](mlir::Operation *op) -> std::string {
    auto sv = readSubview(op);
    if (sv.viewName.empty() || sv.viewName[0] != '%')
      llvm::report_fatal_error("pto.subview destination must be a %name symbol");

    auto baseVar = resolve(sv.baseView);
    auto viewKey = sv.viewName.substr(1);
    auto viewVar = "g_" + viewKey;
    localTensorVars[sv.viewName] = viewVar;
    if (auto it = tensorInfoByResolvedName.find(baseVar); it != tensorInfoByResolvedName.end())
      tensorInfoByResolvedName[viewVar] = it->second;

    std::ostringstream ss;
    ss << "  auto* " << viewVar << "_base = " << baseVar << ".data();\n";
    ss << "  decltype(" << baseVar << ") " << viewVar << "(" << viewVar << "_base);\n";
    ss << "  auto " << viewVar << "_off = (" << resolve(sv.offsets5[0]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_0) + (" << resolve(sv.offsets5[1]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_1) + (" << resolve(sv.offsets5[2]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_2) + (" << resolve(sv.offsets5[3]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_3) + (" << resolve(sv.offsets5[4]) << ") * " << baseVar
       << ".GetStride(GlobalTensorDim::DIM_4);\n";
    ss << "  TASSIGN(" << viewVar << ", " << viewVar << "_base + " << viewVar << "_off);\n";
    return ss.str();
  };

  // Tile address binding (new-format PTO-AS removes explicit `tassign`).
  for (auto &at : allocTiles) {
    if (!at.addrValue)
      continue;
    os << "  TASSIGN(" << resolve(at.tileName) << ", " << resolve(*at.addrValue) << ");\n";
  }
  if (!allocTiles.empty())
    os << "\n";

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

    if (opcode == "idiv") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("idiv expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") / (" + resolve(operands[2]) + ");\n";
    }

    if (opcode == "irem") {
      auto operands = readOperands(op);
      if (operands.size() != 3)
        llvm::report_fatal_error("irem expects 3 operands (dst, src0, src1)");
      auto dst = trim(operands[0]);
      if (!dst.empty() && dst[0] == '%')
        dst = dst.substr(1);
      return "  auto " + dst + " = (" + resolve(operands[1]) + ") % (" + resolve(operands[2]) + ");\n";
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

    if (opcode == "tmatmul_acc") {
      auto operands = readOperands(op);
      if (operands.size() != 4)
        llvm::report_fatal_error("tmatmul_acc expects 4 operands");
      return "  TMATMUL_ACC(" + resolve(operands[0]) + ", " + resolve(operands[1]) + ", " + resolve(operands[2]) +
             ", " + resolve(operands[3]) + ");\n";
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

      auto ti = tensorInfoByResolvedName.find(src);
      auto td = tileDimsByResolvedName.find(dst);
      if (ti != tensorInfoByResolvedName.end() && td != tileDimsByResolvedName.end()) {
        std::ostringstream ss;
        ss << "  // NOTE: tload with indices uses a tile-shaped GlobalTensor view for conversion correctness.\n";
        ss << "  {\n";
        ss << "    auto* " << src << "_ptr = " << src << ".data();\n";
        ss << "    auto " << src << "_off = (" << r0 << ") * " << src
           << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << src
           << ".GetStride(GlobalTensorDim::DIM_4);\n";
        ss << "    using TloadShape = Shape<1, 1, 1, " << td->second.rows << ", " << td->second.cols << ">;\n";
        ss << "    using TloadTensor = GlobalTensor<" << ti->second.elemCpp << ", TloadShape, "
           << ti->second.strideTypeName << ", " << ti->second.layoutCpp << ">;\n";
        ss << "    TloadTensor " << src << "_view(" << src << "_ptr);\n";
        ss << "    TASSIGN(" << src << "_view, " << src << "_ptr + " << src << "_off);\n";
        ss << "    TLOAD(" << dst << ", " << src << "_view);\n";
        ss << "  }\n";
        return ss.str();
      }

      std::ostringstream ss;
      ss << "  // NOTE: tload with indices is lowered via pointer bump (prototype).\n";
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

      auto ti = tensorInfoByResolvedName.find(dst);
      auto td = tileDimsByResolvedName.find(src);
      if (ti != tensorInfoByResolvedName.end() && td != tileDimsByResolvedName.end()) {
        std::ostringstream ss;
        ss << "  // NOTE: tstore with indices uses a tile-shaped GlobalTensor view for conversion correctness.\n";
        ss << "  {\n";
        ss << "    auto* " << dst << "_ptr = " << dst << ".data();\n";
        ss << "    auto " << dst << "_off = (" << r0 << ") * " << dst
           << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
           << ".GetStride(GlobalTensorDim::DIM_4);\n";
        ss << "    using TstoreShape = Shape<1, 1, 1, " << td->second.rows << ", " << td->second.cols << ">;\n";
        ss << "    using TstoreTensor = GlobalTensor<" << ti->second.elemCpp << ", TstoreShape, "
           << ti->second.strideTypeName << ", " << ti->second.layoutCpp << ">;\n";
        ss << "    TstoreTensor " << dst << "_view(" << dst << "_ptr);\n";
        ss << "    TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
        ss << "    TSTORE(" << dst << "_view, " << src << ");\n";
        ss << "  }\n";
        return ss.str();
      }

      std::ostringstream ss;
      ss << "  // NOTE: tstore with indices is lowered via pointer bump (prototype).\n";
      ss << "  auto* " << dst << "_ptr = " << dst << ".data();\n";
      ss << "  decltype(" << dst << ") " << dst << "_view(" << dst << "_ptr);\n";
      ss << "  auto " << dst << "_off = (" << r0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_3) + (" << c0 << ") * " << dst
         << ".GetStride(GlobalTensorDim::DIM_4);\n";
      ss << "  TASSIGN(" << dst << "_view, " << dst << "_ptr + " << dst << "_off);\n";
      ss << "  TSTORE(" << dst << "_view, " << src << ");\n";
      return ss.str();
    }

    // Generic fallback: most PTO ISA ops have a corresponding C++ macro of the same name in uppercase.
    auto operands = readOperands(op);
    if (operands.empty())
      return "  // " + opcode + "\n";
    std::string macro;
    macro.reserve(opcode.size());
    for (char ch : opcode)
      macro.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
    std::ostringstream ss;
    ss << "  " << macro << "(";
    for (size_t i = 0; i < operands.size(); ++i) {
      if (i)
        ss << ", ";
      ss << resolve(operands[i]);
    }
    ss << ");\n";
    return ss.str();
  };

  auto emitBlock = [&](mlir::Block &block, int depth, auto &&self) -> void {
    std::string extra(depth * 2, ' ');

    for (auto &op : block.getOperations()) {
      auto name = op.getName().getStringRef();
      if (name == "pto.subview") {
        os << indentExtra(emitSubviewStmt(&op), extra);
        continue;
      }
      if (name == "pto.arg" || name == "pto.const" || name == "pto.record_event" || name == "pto.wait_event" ||
          name == "pto.tsync" ||
          name == "pto.make_tensor_view" || name == "pto.alloc_tile")
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
