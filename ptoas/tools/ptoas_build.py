#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArgDecl:
    name: str  # without leading %
    type_str: str


@dataclass(frozen=True)
class ConstDecl:
    name: str  # without leading %
    literal: str
    type_str: str


@dataclass(frozen=True)
class Instr:
    opcode: str
    operands: List[str]
    attr_dict: str = ""
    type_sig: str = ""


def _strip_comment(line: str) -> str:
    if ";" in line:
        return line.split(";", 1)[0]
    return line


def _strip_trailing_semicolon(s: str) -> str:
    s = s.strip()
    if s.endswith(";"):
        return s[:-1].rstrip()
    return s


def _split_top_level_commas(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []
    depth_paren = depth_brack = depth_angle = depth_brace = 0
    for ch in s:
        if ch == "(":
            depth_paren += 1
        elif ch == ")":
            depth_paren = max(0, depth_paren - 1)
        elif ch == "[":
            depth_brack += 1
        elif ch == "]":
            depth_brack = max(0, depth_brack - 1)
        elif ch == "<":
            depth_angle += 1
        elif ch == ">":
            depth_angle = max(0, depth_angle - 1)
        elif ch == "{":
            depth_brace += 1
        elif ch == "}":
            depth_brace = max(0, depth_brace - 1)

        if ch == "," and depth_paren == depth_brack == depth_angle == depth_brace == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _split_attr_and_typesig(rest: str) -> Tuple[str, str, str]:
    # Extract optional `{...}` and optional `: ...` (type signature).
    attr_dict = ""
    type_sig = ""
    text = rest.strip()

    if "{" in text:
        # Assume the first {...} block is attr-dict.
        start = text.index("{")
        depth = 0
        end = None
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end is None:
            raise ValueError(f"Unclosed attr dict: {rest}")
        attr_dict = text[start : end + 1].strip()
        text = (text[:start] + " " + text[end + 1 :]).strip()

    if ":" in text:
        before, after = text.split(":", 1)
        text = before.strip()
        type_sig = after.strip()

    return text, attr_dict, type_sig


def _parse_operand(op: str) -> Tuple[str, Optional[Tuple[str, str]]]:
    op = op.strip()
    m = re.fullmatch(r"(%[A-Za-z_][A-Za-z0-9_.]*)\[(.+)\]", op)
    if not m:
        return op, None
    base = m.group(1)
    inside = m.group(2).strip()
    parts = _split_top_level_commas(inside)
    if len(parts) != 2:
        raise ValueError(f"Expected 2 indices in {op}")
    return base, (parts[0].strip(), parts[1].strip())


def parse_pto(path: Path) -> Tuple[List[ArgDecl], List[ConstDecl], List[Instr]]:
    args: List[ArgDecl] = []
    consts: List[ConstDecl] = []
    instrs: List[Instr] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = _strip_trailing_semicolon(_strip_comment(raw)).strip()
        if not line:
            continue

        if line.startswith(".arg "):
            # .arg %x : <type>
            rest = line[len(".arg ") :].strip()
            name_part, type_part = rest.split(":", 1)
            name = name_part.strip()
            if not name.startswith("%"):
                raise ValueError(f".arg name must start with %: {raw}")
            args.append(ArgDecl(name=name[1:], type_str=type_part.strip()))
            continue

        if line.startswith(".const "):
            # .const %c0 = 0 : index
            rest = line[len(".const ") :].strip()
            lhs, type_part = rest.split(":", 1)
            lhs = lhs.strip()
            type_str = type_part.strip()
            name_part, lit_part = lhs.split("=", 1)
            name = name_part.strip()
            lit = lit_part.strip()
            if not name.startswith("%"):
                raise ValueError(f".const name must start with %: {raw}")
            consts.append(ConstDecl(name=name[1:], literal=lit, type_str=type_str))
            continue

        # Instruction: <opcode> <operands...> [attr-dict] [: type_sig]
        opcode, rest = line.split(None, 1)
        rest, attr_dict, type_sig = _split_attr_and_typesig(rest)
        operands = _split_top_level_commas(rest)
        instrs.append(Instr(opcode=opcode.strip(), operands=operands, attr_dict=attr_dict, type_sig=type_sig))

    return args, consts, instrs


def _parse_list_int5(s: str) -> List[str]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"Expected list literal [..]: {s}")
    items = [x.strip() for x in s[1:-1].split(",") if x.strip()]
    if len(items) != 5:
        raise ValueError(f"Expected 5 elements: {s}")
    return items


def _cpp_int(v: str) -> str:
    v = v.strip()
    if v.lower() == "dyn":
        return "pto::DYNAMIC"
    return v


def _cpp_layout(v: str) -> str:
    return f"Layout::{v.strip()}"


def _cpp_tiletype(v: str) -> str:
    return f"TileType::{v.strip()}"


def _cpp_blayout(v: str) -> str:
    return f"BLayout::{v.strip()}"


def _cpp_slayout(v: str) -> str:
    return f"SLayout::{v.strip()}"


def _cpp_pad(v: str) -> str:
    return f"PadValue::{v.strip()}"


def _cpp_elem(v: str) -> str:
    v = v.strip()
    if v in ("f16", "bf16"):
        return "half"
    if v == "f32":
        return "float"
    if v == "i32":
        return "int32_t"
    if v == "u32":
        return "uint32_t"
    raise ValueError(f"Unsupported element type: {v}")


def _parse_tensor_type(type_str: str) -> Dict[str, str]:
    # Accept:
    # !pto.tensor<dtype=f16, shape=[...], stride=[...], layout=ND>
    # (compat) !pto.gtensor<element=f16, ...>
    compact = type_str.replace(" ", "")
    m = re.fullmatch(r"!pto\.(?:tensor|gtensor)<(.+)>", compact)
    if not m:
        raise ValueError(f"Not a tensor type: {type_str}")
    inner = m.group(1)
    parts = [p.strip() for p in _split_top_level_commas(inner) if p.strip()]
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=", 1)
            kv[k.strip()] = v.strip()
        else:
            # tolerate shorthand: first token is element type
            kv.setdefault("dtype", p.strip())
    # compat: element -> dtype
    if "dtype" not in kv and "element" in kv:
        kv["dtype"] = kv["element"]
    for k in ("dtype", "shape", "stride", "layout"):
        if k not in kv:
            raise ValueError(f"tensor missing {k}: {type_str}")
    return kv


def _parse_tile_type(type_str: str) -> Dict[str, str]:
    m = re.fullmatch(r"!pto\.tile<(.+)>", type_str.replace(" ", ""))
    if not m:
        raise ValueError(f"Not a tile type: {type_str}")
    inner = m.group(1)
    parts = [p.strip() for p in _split_top_level_commas(inner) if p.strip()]
    kv: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    # compat: element -> dtype
    if "dtype" not in kv and "element" in kv:
        kv["dtype"] = kv["element"]
    for k in ("dtype", "rows", "cols"):
        if k not in kv:
            raise ValueError(f"tile missing {k}: {type_str}")
    kv.setdefault("loc", "Vec")
    kv.setdefault("blayout", "RowMajor")
    kv.setdefault("slayout", "NoneBox")
    kv.setdefault("fractal", "512")
    kv.setdefault("pad", "Null")
    kv.setdefault("valid", f"{kv['rows']}x{kv['cols']}")
    return kv


def emit_cpp(
    *,
    kernel_name: str,
    args: List[ArgDecl],
    consts: List[ConstDecl],
    instrs: List[Instr],
    memory_model: str,
    repo_root: Path,
) -> str:
    tensor_args: List[Tuple[ArgDecl, Dict[str, str]]] = []
    tile_locals: List[Tuple[ArgDecl, Dict[str, str]]] = []

    for a in args:
        if a.type_str.startswith("!pto.tensor<") or a.type_str.startswith("!pto.gtensor<"):
            tensor_args.append((a, _parse_tensor_type(a.type_str)))
        elif a.type_str.startswith("!pto.tile<"):
            tile_locals.append((a, _parse_tile_type(a.type_str)))

    name_to_cpp: Dict[str, str] = {}
    tensor_cpp_obj: Dict[str, str] = {}
    tile_cpp_obj: Dict[str, str] = {}

    # Kernel signature uses only tensor args (GM_ADDR pointers).
    params: List[str] = []
    for a, _ in tensor_args:
        params.append(f"GM_ADDR {a.name}")
        tensor_cpp_obj[a.name] = f"g_{a.name}"
        name_to_cpp[a.name] = tensor_cpp_obj[a.name]

    for a, _ in tile_locals:
        tile_cpp_obj[a.name] = f"t_{a.name}"
        name_to_cpp[a.name] = tile_cpp_obj[a.name]

    for c in consts:
        name_to_cpp[c.name] = f"c_{c.name}"

    def resolve_value(v: str) -> str:
        v = v.strip()
        if v.startswith("%"):
            key = v[1:]
            if key not in name_to_cpp:
                raise ValueError(f"Unknown value: {v}")
            return name_to_cpp[key]
        return v

    lines: List[str] = []
    lines.append("// Generated by ptoas/tools/ptoas_build.py")
    lines.append(f"#define {memory_model}")
    lines.append("#include \"kernel_operator.h\"")
    lines.append("#include <pto/pto-inst.hpp>")
    lines.append("#include <cstdint>")
    lines.append("using namespace pto;")
    lines.append("")
    lines.append(f"extern \"C\" __global__ AICORE void {kernel_name}({', '.join(params)}) {{")

    # GlobalTensor declarations.
    for a, t in tensor_args:
        elem_cpp = _cpp_elem(t["dtype"])
        shape = _parse_list_int5(t["shape"])
        stride = _parse_list_int5(t["stride"])
        layout = t["layout"]
        lines.append(f"  using {a.name}_Shape = Shape<{', '.join(_cpp_int(x) for x in shape)}>;")
        lines.append(f"  using {a.name}_Stride = Stride<{', '.join(_cpp_int(x) for x in stride)}>;")
        lines.append(
            f"  using {a.name}_GT = GlobalTensor<{elem_cpp}, {a.name}_Shape, {a.name}_Stride, {_cpp_layout(layout)}>;"
        )
        lines.append(f"  {a.name}_GT {tensor_cpp_obj[a.name]}((__gm__ {elem_cpp}*){a.name});")
        lines.append("")

    # Tile declarations.
    for a, t in tile_locals:
        elem_cpp = _cpp_elem(t["dtype"])
        rows = _cpp_int(t["rows"])
        cols = _cpp_int(t["cols"])
        loc = _cpp_tiletype(t["loc"])
        blayout = _cpp_blayout(t["blayout"])
        slayout = _cpp_slayout(t["slayout"])
        fractal = _cpp_int(t["fractal"])
        pad = _cpp_pad(t["pad"])
        valid = t["valid"]
        if "x" not in valid:
            raise ValueError(f"tile valid must be RxC: {t}")
        vrow, vcol = valid.split("x", 1)
        vrow_cpp = _cpp_int(vrow)
        vcol_cpp = _cpp_int(vcol)
        lines.append(
            f"  using {a.name}_Tile = Tile<{loc}, {elem_cpp}, {rows}, {cols}, {blayout}, {vrow_cpp}, {vcol_cpp}, {slayout}, {fractal}, {pad}>;"
        )
        lines.append(f"  {a.name}_Tile {tile_cpp_obj[a.name]};")
    if tile_locals:
        lines.append("")

    # Constants.
    for c in consts:
        # Keep as uint64_t for address constants and indices.
        lit = c.literal
        if lit.startswith("0x") or lit.startswith("0X"):
            lines.append(f"  constexpr uint64_t c_{c.name} = {lit};")
        else:
            lines.append(f"  constexpr int64_t c_{c.name} = {lit};")
    if consts:
        lines.append("")

    def emit_call(opcode: str, ops: List[str]) -> List[str]:
        op = opcode.lower()
        if op == "tassign":
            if len(ops) != 2:
                raise ValueError(f"tassign expects 2 operands, got {ops}")
            return [f"  TASSIGN({resolve_value(ops[0])}, {resolve_value(ops[1])});"]

        if op == "tadd":
            if len(ops) != 3:
                raise ValueError(f"tadd expects 3 operands, got {ops}")
            return [f"  TADD({resolve_value(ops[0])}, {resolve_value(ops[1])}, {resolve_value(ops[2])});"]

        if op == "tload":
            if len(ops) != 2:
                raise ValueError(f"tload expects 2 operands, got {ops}")
            base, idx = _parse_operand(ops[1])
            if idx is None:
                return [f"  TLOAD({resolve_value(ops[0])}, {resolve_value(base)});"]
            # MVP: ignore indices unless they are compile-time zeros.
            r0, c0 = idx
            if resolve_value(r0) in ("0", "c_r0") and resolve_value(c0) in ("0", "c_c0"):
                return [f"  TLOAD({resolve_value(ops[0])}, {resolve_value(base)});"]
            return [
                f"  // NOTE: tload with non-zero indices is lowered via pointer bump (prototype).",
                f"  auto {resolve_value(base)}_view = {resolve_value(base)};",
                f"  auto* {resolve_value(base)}_ptr = {resolve_value(base)}.data();",
                f"  auto {resolve_value(base)}_off = ({resolve_value(r0)}) * {resolve_value(base)}.GetStride(GlobalTensorDim::DIM_3)"
                f" + ({resolve_value(c0)}) * {resolve_value(base)}.GetStride(GlobalTensorDim::DIM_4);",
                f"  TASSIGN({resolve_value(base)}_view, {resolve_value(base)}_ptr + {resolve_value(base)}_off);",
                f"  TLOAD({resolve_value(ops[0])}, {resolve_value(base)}_view);",
            ]

        if op == "tstore":
            if len(ops) != 2:
                raise ValueError(f"tstore expects 2 operands, got {ops}")
            base, idx = _parse_operand(ops[0])
            if idx is None:
                return [f"  TSTORE({resolve_value(base)}, {resolve_value(ops[1])});"]
            r0, c0 = idx
            if resolve_value(r0) in ("0", "c_r0") and resolve_value(c0) in ("0", "c_c0"):
                return [f"  TSTORE({resolve_value(base)}, {resolve_value(ops[1])});"]
            return [
                f"  // NOTE: tstore with non-zero indices is lowered via pointer bump (prototype).",
                f"  auto {resolve_value(base)}_view = {resolve_value(base)};",
                f"  auto* {resolve_value(base)}_ptr = {resolve_value(base)}.data();",
                f"  auto {resolve_value(base)}_off = ({resolve_value(r0)}) * {resolve_value(base)}.GetStride(GlobalTensorDim::DIM_3)"
                f" + ({resolve_value(c0)}) * {resolve_value(base)}.GetStride(GlobalTensorDim::DIM_4);",
                f"  TASSIGN({resolve_value(base)}_view, {resolve_value(base)}_ptr + {resolve_value(base)}_off);",
                f"  TSTORE({resolve_value(base)}_view, {resolve_value(ops[1])});",
            ]

        if op == "mgather":
            if len(ops) != 3:
                raise ValueError(f"mgather expects 3 operands, got {ops}")
            return [f"  MGATHER({resolve_value(ops[0])}, {resolve_value(ops[1])}, {resolve_value(ops[2])});"]

        if op == "mscatter":
            if len(ops) != 3:
                raise ValueError(f"mscatter expects 3 operands, got {ops}")
            return [f"  MSCATTER({resolve_value(ops[0])}, {resolve_value(ops[1])}, {resolve_value(ops[2])});"]

        if op == "tsync":
            return ["  // tsync ignored in synchronous prototype"]

        raise ValueError(f"Unsupported opcode in prototype: {opcode}")

    for ins in instrs:
        for stmt in emit_call(ins.opcode, ins.operands):
            lines.append(stmt)

    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def _ascend_include_dirs(ascend_home: Path) -> List[str]:
    # Keep this list conservative and self-contained; these paths cover the
    # `kernel_operator.h` include chain on common toolkit layouts.
    candidates = [
        ascend_home / "compiler/ascendc/include/basic_api",
        ascend_home / "compiler/ascendc/include/basic_api/impl",
        ascend_home / "compiler/asc/include/basic_api",
        ascend_home / "compiler/asc/include/interface",
        ascend_home / "compiler/asc",
        ascend_home / "include/ascendc",
        ascend_home / "include",
        ascend_home / "runtime/include",
    ]
    out: List[str] = []
    for p in candidates:
        if p.exists():
            out.append(str(p))
    return out


def build_bin(
    *,
    cpp_path: Path,
    out_obj: Path,
    out_bin: Path,
    arch: str,
    ascend_home: Path,
    repo_root: Path,
) -> None:
    bisheng = shutil.which("bisheng")
    if not bisheng:
        raise RuntimeError("bisheng not found in PATH; source Ascend setenv.bash first.")

    include_dirs = _ascend_include_dirs(ascend_home) + [str(repo_root / "include")]

    cmd = [
        bisheng,
        "-xcce",
        f"--cce-aicore-arch={arch}",
        "-std=c++17",
    ]
    for inc in include_dirs:
        cmd.append(f"-I{inc}")
    cmd += ["-c", str(cpp_path), "-o", str(out_obj)]

    subprocess.run(cmd, check=True)

    objcopy = shutil.which("objcopy")
    if not objcopy:
        raise RuntimeError("objcopy not found in PATH")
    subprocess.run(
        [objcopy, f"--dump-section", f"__aicore_rel_binary={out_bin}", str(out_obj)],
        check=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Prototype PTO-AS -> AscendC -> bin pipeline (no MLIR required).")
    ap.add_argument("input", type=Path, help="Input PTO-AS file (*.pto)")
    ap.add_argument("--outdir", type=Path, default=Path("ptoas/out"), help="Output directory")
    ap.add_argument("--arch", default="dav-c220-vec", help="CCE aicore arch (e.g. dav-c220-vec, dav-c220-cube, dav-c310)")
    ap.add_argument("--memory-model", default="MEMORY_BASE", choices=["MEMORY_BASE", "REGISTER_BASE"])
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    ascend_home_env = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home_env:
        raise SystemExit("ASCEND_HOME_PATH is not set; source Ascend setenv.bash first.")
    ascend_home = Path(ascend_home_env).resolve()

    args.outdir.mkdir(parents=True, exist_ok=True)
    kernel_name = args.input.stem

    decl_args, decl_consts, instrs = parse_pto(args.input)
    cpp_text = emit_cpp(
        kernel_name=kernel_name,
        args=decl_args,
        consts=decl_consts,
        instrs=instrs,
        memory_model=args.memory_model,
        repo_root=repo_root,
    )

    cpp_path = args.outdir / f"{kernel_name}_kernel.cpp"
    obj_path = args.outdir / f"{kernel_name}_kernel.o"
    bin_path = args.outdir / f"{kernel_name}.bin"
    cpp_path.write_text(cpp_text, encoding="utf-8")

    build_bin(
        cpp_path=cpp_path,
        out_obj=obj_path,
        out_bin=bin_path,
        arch=args.arch,
        ascend_home=ascend_home,
        repo_root=repo_root,
    )

    print(f"Wrote: {cpp_path}")
    print(f"Wrote: {obj_path}")
    print(f"Wrote: {bin_path}")


if __name__ == "__main__":
    main()
