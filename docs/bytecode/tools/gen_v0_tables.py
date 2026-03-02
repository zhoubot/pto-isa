#!/usr/bin/env python3
"""Generate PTO-BC v0 opcode tables and Sail opcode schema.

Inputs (versioned in repo):
- docs/isa/manifest.yaml            (ISA instruction list)
- docs/ir/PTO-IR-ops.md             (non-ISA PTO-IR ops)
- docs/bytecode/samples/*.pto       (PTODSL-emitted samples; used to ensure coverage)

Outputs:
- docs/bytecode/generated/opcodes_v0.md
- docs/bytecode/generated/op_families_v0.json
- tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail
- tools/ptobc/generated/ptobc_opcodes_v0.h

NOTE:
- This generator assigns *stable* opcodes by sorting op names within each segment.
- Goal: every frozen opcode in v0 has a compact schema entry so Sail can decode/validate without GENERIC_OP.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

# __file__ = <repo>/docs/bytecode/tools/gen_v0_tables.py
# parents: [tools, bytecode, docs, <repo>]
ROOT = Path(__file__).resolve().parents[3]

ISA_MANIFEST = ROOT / "docs/isa/manifest.yaml"
IR_OPS_DOC = ROOT / "docs/ir/PTO-IR-ops.md"
SAMPLES_DIR = ROOT / "docs/bytecode/samples"

OUT_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
OUT_FAMILIES = ROOT / "docs/bytecode/generated/op_families_v0.json"
OUT_SAIL = ROOT / "tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail"
OUT_ISA_REPORT = ROOT / "docs/bytecode/generated/isa_dps_arity_v0.md"
OUT_PTOBC_HDR = ROOT / "tools/ptobc/generated/ptobc_opcodes_v0.h"

# Opcode segments (u16)
SEG_PTO_L1 = 0x0000
SEG_PTO_L2 = 0x1000
SEG_ARITH = 0x2000
SEG_SCF = 0x4000
SEG_OTHER = 0x6000

# Families: base -> known variants (u8 values will be assigned by sorted order)
#
# IMPORTANT:
# - Some families have a "base" form with no suffix (e.g. `pto.tmatmul`).
#   For those, we reserve variant_u8=0 to mean "base" and start explicit suffix variants from 1.
# - Stage6 matmul/gemv families are *discovered from PTOAS TableGen* to ensure perfect alignment.

FAMILIES = {
    # section kind is mandatory (no base variant)
    "pto.section": ["cube", "vector"],
}

FAMILY_HAS_BASE_VARIANT = {
    "pto.section": False,
}

# PTO ops documented as L1/L2 in docs/ir/PTO-IR-ops.md (best effort)
# If not found here, we treat pto.t*/pto.m* ISA-ish ops as L2.
PTO_L1_NAMES = {
    "pto.make_tensor_view",
    "pto.partition_view",
    "pto.get_block_idx",
    "pto.get_subblock_idx",
    "pto.get_block_num",
    "pto.get_subblock_num",
    "pto.section",  # family
}

PTO_L2_NAMES = {
    "pto.alloc_tile",
    "pto.addptr",
    "pto.tgetval",
    "pto.tsetval",
    "pto.record_event",
    "pto.wait_event",
    "pto.barrier",
}

# Some textual ops appear without dialect prefix in MLIR assembly.
# We normalize them here.
ALIAS_OPS = {
    "return": "func.return",
}

OP_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b")

ISA_DOCS_DIR = ROOT / "docs/isa"
PTOOPS_TD = ROOT / "PTOAS/include/PTO/IR/PTOOps.td"


def parse_pto_ops_tablegen() -> dict[str, dict]:
    """Parse PTOAS TableGen ops for operand arity.

    Returns: op_name ("pto.<name>") -> {
      min_operands: int,
      has_variadic: bool,
      has_optional: bool,
      operands: list[{name, is_optional, is_variadic}],
    }

    Notes:
    - Attributes declared in `arguments = (ins ...)` (e.g. OptionalAttr<...>:$layout, TileBufConfigAttr:$config)
      are excluded from operands.
    - Variadic operands contribute min 0.
    - Optional operands contribute min 0.
    """
    txt = PTOOPS_TD.read_text(encoding="utf-8")

    # Split into `def ...` blocks.
    blocks = re.split(r"\n(?=def\s+)", txt)

    op_pat = re.compile(r"def\s+\w+\s*:\s*PTO_[A-Za-z0-9_]*Op<\"([^\"]+)\"")
    args_pat = re.compile(r"let\s+arguments\s*=\s*\(ins(?P<body>.*?)\);", re.S)
    entry_pat = re.compile(r"(?P<ty>[^:()\n]+):\$(?P<name>[A-Za-z0-9_]+)")

    out: dict[str, dict] = {}

    def is_attr_type(ty: str) -> bool:
        ty = ty.strip()
        if "OptionalAttr<" in ty:
            return True
        # Bare ...Attr or ...Attr<...>
        if re.search(r"\b[A-Za-z0-9_]*Attr\b", ty):
            return True
        if "Attr<" in ty:
            return True
        return False

    for blk in blocks:
        m = op_pat.search(blk)
        if not m:
            continue
        opname = m.group(1)
        op_full = f"pto.{opname}"

        am = args_pat.search(blk)
        if not am:
            continue
        body = am.group("body")

        operands = []
        has_var = False
        has_opt = False
        min_ops = 0

        for em in entry_pat.finditer(body):
            ty = em.group("ty").strip()
            nm = em.group("name")

            if is_attr_type(ty):
                continue

            is_var = "Variadic<" in ty
            is_opt = "Optional<" in ty

            operands.append({"name": nm, "is_optional": is_opt, "is_variadic": is_var})

            if is_var:
                has_var = True
            if is_opt:
                has_opt = True

            if (not is_var) and (not is_opt):
                min_ops += 1

        out[op_full] = {
            "min_operands": min_ops,
            "has_variadic": has_var,
            "has_optional": has_opt,
            "operands": operands,
        }

    return out


def _count_dps_args(seg: str) -> int:
    seg = seg.strip()
    if not seg:
        return 0
    # Drop type suffix after ':'
    seg = seg.split(":", 1)[0]
    seg = seg.strip().rstrip(")")
    if seg.strip() == "":
        return 0
    return sum(1 for t in (x.strip() for x in seg.split(",")) if t)


def build_dps_arity_index() -> dict[str, int]:
    """Best-effort map: pto.op_name -> (ins_count + outs_count) from docs/isa."""
    out: dict[str, int] = {}
    for md in sorted(ISA_DOCS_DIR.glob("*.md")):
        txt = md.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if "pto." not in line or "ins(" not in line or "outs(" not in line:
                continue
            # Extract op token
            m = re.search(r"\b(pto\.[A-Za-z0-9_.]+)\b", line)
            if not m:
                continue
            op = m.group(1)
            # Pull ins(...) and outs(...)
            mi = re.search(r"ins\(([^)]*)\)", line)
            mo = re.search(r"outs\(([^)]*)\)", line)
            if not mi or not mo:
                continue
            ins_n = _count_dps_args(mi.group(1))
            outs_n = _count_dps_args(mo.group(1))
            ar = ins_n + outs_n
            out[op] = max(out.get(op, 0), ar)
    return out


def load_isa_ops() -> set[str]:
    """ISA ops for opcode freeze.

    We take the union of:
    - all DPS-form PTO op spellings discovered in docs/isa (ins/outs)
    - a small set of known non-DPS ops referenced by docs/samples

    This avoids accidentally treating type/attr spellings (e.g. `#pto.op<TADD>`) as ops.
    """
    dps = build_dps_arity_index()
    ops = set(dps.keys())
    # Non-DPS, but real PTO IR ops used in `.pto`.
    ops.add("pto.tsync")
    return ops


def parse_ir_doc_ops() -> set[str]:
    txt = IR_OPS_DOC.read_text(encoding="utf-8")
    ops = set()
    for m in OP_RE.finditer(txt):
        # Filter out:
        # - type spellings like `!pto.tensor_view<...>`
        # - attribute spellings like `#pto.event<...>`
        # - quoted attribute keys like "pto.device-spec"
        start = m.start()
        if start > 0 and txt[start - 1] in {'!', '"', '#'}:
            continue

        dial = m.group(1)
        op = m.group(2)
        if dial in {"pto", "arith", "scf", "func"}:
            ops.add(f"{dial}.{op}")
    return ops


def parse_sample_ops() -> set[str]:
    ops = set()
    for p in sorted(SAMPLES_DIR.glob("*.pto")):
        txt = p.read_text(encoding="utf-8")
        for raw in txt.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            # Handle bare 'return'
            if line.startswith("return"):
                ops.add("func.return")
                continue

            # Find all dial.op-like tokens, but filter out:
            # - type spellings like `!pto.ptr<...>` / `!pto.tensor_view<...>`
            # - attribute keys like "pto.device-spec"
            for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+", raw):
                start = m.start()
                if start > 0 and raw[start - 1] in {'!', '"', '#'}:
                    continue
                name = m.group(0)
                if name.startswith("pto.section."):
                    ops.add("pto.section")
                else:
                    ops.add(name)
    return ops


def normalize_op_name(name: str) -> tuple[str, str | None]:
    """Return (base_name, variant_or_none).

    Variant is a *string label* (e.g. "acc", "bias", "vector").
    For base-form ops with no suffix, variant is None.

    Matching uses **longest-base first** to avoid swallowing nested families
    (e.g. `pto.tmatmul.mx.*` should match base `pto.tmatmul.mx`, not `pto.tmatmul`).
    """
    if name in ALIAS_OPS:
        name = ALIAS_OPS[name]

    # Exact family base op (e.g. `pto.tmatmul.mx`) should stay as base.
    if name in FAMILIES:
        return name, None

    # Special-case section kind: pto.section.vector/cube
    if name.startswith("pto.section."):
        parts = name.split(".")
        if len(parts) >= 3:
            return "pto.section", parts[2]

    for base in sorted(FAMILIES.keys(), key=len, reverse=True):
        if name.startswith(base + "."):
            suf = name[len(base) + 1 :]
            return base, suf

    return name, None


def classify_segment(base: str) -> tuple[str, int]:
    """Return (segment_name, segment_base)."""
    if base.startswith("pto."):
        if base in PTO_L1_NAMES:
            return ("pto-l1", SEG_PTO_L1)
        if base in PTO_L2_NAMES:
            return ("pto-l2", SEG_PTO_L2)
        # heuristic: ISA-ish tile ops are L2
        if base.startswith("pto.t") or base.startswith("pto.m"):
            return ("pto-l2", SEG_PTO_L2)
        # fallback
        return ("pto-l1", SEG_PTO_L1)
    if base.startswith("arith."):
        return ("arith", SEG_ARITH)
    if base.startswith("scf."):
        return ("scf", SEG_SCF)
    # func/other dialects
    return ("other", SEG_OTHER)


@dataclass(frozen=True)
class OpcodeRow:
    opcode: int
    segment: str
    op: str


def assign_opcodes(ops: Iterable[str]) -> tuple[list[OpcodeRow], dict[str, dict[str, int]]]:
    base_ops: set[str] = set()
    families: dict[str, set[str]] = {k: set() for k in FAMILIES}

    for op in ops:
        base, var = normalize_op_name(op)
        base_ops.add(base)
        if var is not None and base in families:
            families[base].add(var)

    # Ensure all declared family variants exist in table
    for base, vs in FAMILIES.items():
        base_ops.add(base)
        families.setdefault(base, set()).update(vs)

        # Add implicit base variant for families that have a base form.
        if FAMILY_HAS_BASE_VARIANT.get(base, True):
            families[base].add("base")

    # Group by segment
    seg_map: dict[tuple[str, int], list[str]] = {}
    for base in base_ops:
        seg = classify_segment(base)
        seg_map.setdefault(seg, []).append(base)

    rows: list[OpcodeRow] = []
    for (seg_name, seg_base), names in sorted(seg_map.items(), key=lambda x: x[0][1]):
        names_sorted = sorted(set(names))
        for i, name in enumerate(names_sorted):
            rows.append(OpcodeRow(opcode=seg_base + i, segment=seg_name, op=name))

    # Assign variant u8 values per family
    fam_map: dict[str, dict[str, int]] = {}
    for base, vars_ in families.items():
        m: dict[str, int] = {}
        has_base = FAMILY_HAS_BASE_VARIANT.get(base, True)

        if has_base and "base" in vars_:
            m["base"] = 0
            others = sorted(v for v in vars_ if v != "base")
            for i, v in enumerate(others, start=1):
                m[v] = i
        else:
            others = sorted(vars_)
            for i, v in enumerate(others):
                m[v] = i

        fam_map[base] = m

    return rows, fam_map


def write_md(rows: list[OpcodeRow], fam_map: dict[str, dict[str, int]]):
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# PTO-BC v0 — Opcode table (generated)\n")
    lines.append("Generated from: docs/isa/manifest.yaml, docs/ir/PTO-IR-ops.md, docs/bytecode/samples/*.pto\n")
    lines.append("\n")
    lines.append("## Families (variant u8)\n")
    for base, m in sorted(fam_map.items()):
        lines.append(f"- `{base}`: " + ", ".join([f"{k}={v}" for k, v in sorted(m.items(), key=lambda x: x[1])]) + "\n")
    lines.append("\n")
    lines.append("## Opcodes\n")
    lines.append("- Format: `0xXXXX  segment  op`\n\n")
    for r in sorted(rows, key=lambda x: x.opcode):
        lines.append(f"- `0x{r.opcode:04X}`  {r.segment:<7}  `{r.op}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")


def write_families_json(fam_map: dict[str, dict[str, int]]):
    OUT_FAMILIES.parent.mkdir(parents=True, exist_ok=True)
    OUT_FAMILIES.write_text(json.dumps(fam_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_isa_report(rows: list[OpcodeRow], schema_entries: dict[int, dict], dps: dict[str, int]):
    op_to_opcode = {r.op: r.opcode for r in rows}

    lines = []
    lines.append("# PTO-BC v0 — ISA DPS arity report (generated)\n\n")
    lines.append("Source: `docs/isa/*.md` DPS lines `pto.xxx ins(...) outs(...)`.\n\n")
    lines.append("Columns: op | dps_arity | operand_mode | num_operands | imm_kind\n\n")

    def mode_name(x: int) -> str:
        return {
            0: "fixed",
            1: "by_variant",
            2: "varcount",
            3: "segmented",
            4: "optmask2",
        }.get(x, f"0x{x:02X}")

    for op in sorted(dps.keys()):
        opc = op_to_opcode.get(op)
        if opc is None:
            lines.append(f"- `{op}`: MISSING OPCODE\n")
            continue
        sch = schema_entries.get(opc)
        if not sch:
            lines.append(f"- `{op}`: MISSING SCHEMA\n")
            continue
        lines.append(
            f"- `{op}` | {dps[op]} | {mode_name(sch['operand_mode'])} | {sch['num_operands']} | 0x{sch['imm_kind']:02X}\n"
        )

    OUT_ISA_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_ISA_REPORT.write_text("".join(lines), encoding="utf-8")


def write_ptobc_header(rows: list[OpcodeRow],
                       fam_map: dict[str, dict[str, int]],
                       schema_entries: dict[int, dict],
                       by_variant_entries: dict[int, dict[int, int]]):
    """Emit a C++ header for ptobc with name/opcode/schema lookups."""

    OUT_PTOBC_HDR.parent.mkdir(parents=True, exist_ok=True)

    # Build opcode -> op name map.
    opcode_to_name = {r.opcode: r.op for r in rows}

    lines: list[str] = []
    lines.append("// Generated by docs/bytecode/tools/gen_v0_tables.py\n")
    lines.append("#pragma once\n\n")
    lines.append("#include <cstdint>\n")
    lines.append("#include <optional>\n\n")
    lines.append("#include <llvm/ADT/StringRef.h>\n")
    lines.append("#include <llvm/ADT/StringSwitch.h>\n\n")

    lines.append("namespace ptobc::v0 {\n\n")
    lines.append("struct OpInfo {\n")
    lines.append("  uint16_t opcode;\n")
    lines.append("  const char *name;\n")
    lines.append("  uint8_t has_variant_u8;\n")
    lines.append("  uint8_t result_type_mode;\n")
    lines.append("  uint8_t operand_mode;\n")
    lines.append("  uint16_t num_operands;\n")
    lines.append("  uint16_t num_results;\n")
    lines.append("  uint16_t num_regions;\n")
    lines.append("  uint8_t imm_kind;\n")
    lines.append("};\n\n")

    # Canonical array sorted by opcode.
    lines.append("inline constexpr OpInfo kOpTable[] = {\n")
    for opc in sorted(schema_entries.keys()):
        sch = schema_entries[opc]
        name = opcode_to_name.get(opc)
        if name is None:
            raise RuntimeError(f"missing opname for opcode 0x{opc:04X}")
        lines.append(
            "  {0x%04X, \"%s\", %d, 0x%02X, 0x%02X, %d, %d, %d, 0x%02X},\n"
            % (
                opc,
                name,
                1 if sch["has_variant_u8"] else 0,
                sch["result_type_mode"],
                sch["operand_mode"],
                sch["num_operands"],
                sch["num_results"],
                sch["num_regions"],
                sch["imm_kind"],
            )
        )
    lines.append("};\n\n")

    lines.append("inline const OpInfo *lookupByOpcode(uint16_t opcode) {\n")
    lines.append("  // Binary search on kOpTable (sorted by opcode).\n")
    lines.append("  size_t lo = 0, hi = sizeof(kOpTable) / sizeof(kOpTable[0]);\n")
    lines.append("  while (lo < hi) {\n")
    lines.append("    size_t mid = lo + (hi - lo) / 2;\n")
    lines.append("    uint16_t v = kOpTable[mid].opcode;\n")
    lines.append("    if (v == opcode) return &kOpTable[mid];\n")
    lines.append("    if (v < opcode) lo = mid + 1; else hi = mid;\n")
    lines.append("  }\n")
    lines.append("  return nullptr;\n")
    lines.append("}\n\n")

    # name -> opcode switch.
    lines.append("inline std::optional<uint16_t> lookupOpcodeByName(llvm::StringRef name) {\n")
    lines.append("  uint16_t v = llvm::StringSwitch<uint16_t>(name)\n")
    for r in sorted(rows, key=lambda x: x.op):
        lines.append(f"    .Case(\"{r.op}\", 0x{r.opcode:04X})\n")
    lines.append("    .Default(0xFFFF);\n")
    lines.append("  if (v == 0xFFFF) return std::nullopt;\n")
    lines.append("  return v;\n")
    lines.append("}\n\n")

    lines.append("inline const OpInfo *lookupByName(llvm::StringRef name) {\n")
    lines.append("  auto o = lookupOpcodeByName(name);\n")
    lines.append("  if (!o) return nullptr;\n")
    lines.append("  return lookupByOpcode(*o);\n")
    lines.append("}\n\n")

    # Full op-name -> (opcode, variant_u8) mapping (families expanded).
    lines.append("struct OpcodeAndVariant { uint16_t opcode; uint8_t hasVariant; uint8_t variant; };\n\n")

    lines.append("inline std::optional<OpcodeAndVariant> lookupOpcodeAndVariantByFullName(llvm::StringRef fullName) {\n")
    lines.append("  // For non-family ops, variant is 0. For family ops, variant is the assigned u8.\n")
    lines.append("  // NOTE: `pto.section` is not a real op name; use `pto.section.cube`/`pto.section.vector`.\n")
    lines.append("  return llvm::StringSwitch<std::optional<OpcodeAndVariant>>(fullName)\n")

    for r in sorted(rows, key=lambda x: x.op):
        if r.op in fam_map:
            continue
        lines.append(f"    .Case(\"{r.op}\", OpcodeAndVariant{{0x{r.opcode:04X}, 0, 0}})\n")

    for base, vm in sorted(fam_map.items()):
        base_opc = next((rr.opcode for rr in rows if rr.op == base), None)
        if base_opc is None:
            raise RuntimeError(f"missing opcode for family base {base}")
        for label, vid in sorted(vm.items(), key=lambda x: x[1]):
            full = base if label == "base" else f"{base}.{label}"
            if base == "pto.section" and label == "base":
                continue
            lines.append(f"    .Case(\"{full}\", OpcodeAndVariant{{0x{base_opc:04X}, 1, {vid}}})\n")

    lines.append("    .Default(std::nullopt);\n")
    lines.append("}\n\n")

    lines.append("inline const char *fullNameFromOpcodeVariant(uint16_t opcode, uint8_t variant) {\n")
    lines.append("  const OpInfo *info = lookupByOpcode(opcode);\n")
    lines.append("  if (!info) return nullptr;\n")
    lines.append("  if (!info->has_variant_u8) return info->name;\n")
    lines.append("  switch (opcode) {\n")
    for base, vm in sorted(fam_map.items()):
        base_opc = next((rr.opcode for rr in rows if rr.op == base), None)
        if base_opc is None:
            raise RuntimeError(f"missing opcode for family base {base}")
        lines.append(f"  case 0x{base_opc:04X}:\n")
        lines.append("    switch (variant) {\n")
        for label, vid in sorted(vm.items(), key=lambda x: x[1]):
            full = base if label == "base" else f"{base}.{label}"
            if base == "pto.section" and label == "base":
                continue
            lines.append(f"    case {vid}: return \"{full}\";\n")
        lines.append("    default: return info->name;\n")
        lines.append("    }\n")
    lines.append("  default: return info->name;\n")
    lines.append("  }\n")
    lines.append("}\n\n")

    # by-variant operand count lookup.
    lines.append("inline std::optional<int> lookupOperandsByVariant(uint16_t opcode, uint8_t variant) {\n")
    lines.append("  switch (opcode) {\n")
    for opc in sorted(by_variant_entries.keys()):
        vm = by_variant_entries[opc]
        lines.append(f"  case 0x{opc:04X}:\n")
        lines.append("    switch (variant) {\n")
        for vid in sorted(vm.keys()):
            lines.append(f"    case {vid}: return {vm[vid]};\n")
        lines.append("    default: return std::nullopt;\n")
        lines.append("    }\n")
    lines.append("  default: return std::nullopt;\n")
    lines.append("  }\n")
    lines.append("}\n\n")

    # Variant label maps (for future encoder use).
    lines.append("// Variant maps (label -> variant_u8), mirrored from the Sail generator:\n")
    lines.append("// ")
    lines.append(json.dumps(fam_map, sort_keys=True))
    lines.append("\n\n")

    lines.append("} // namespace ptobc::v0\n")

    OUT_PTOBC_HDR.write_text("".join(lines), encoding="utf-8")


def write_sail(rows: list[OpcodeRow], fam_map: dict[str, dict[str, int]]):
    """Emit a full opcode schema mapping (v0) for Sail decode/validation."""

    OUT_SAIL.parent.mkdir(parents=True, exist_ok=True)

    dps = build_dps_arity_index()
    tg = parse_pto_ops_tablegen()

    def tg_arity(op: str) -> int | None:
        e = tg.get(op)
        if not e:
            return None
        # For compact fixed-arity schema we require no optional/variadic operands.
        if e.get("has_variadic") or e.get("has_optional"):
            return None
        return int(e.get("min_operands", 0))

    # Helpers: opcode properties
    ARITH_ARITY = {
        "arith.constant": 0,
        "arith.addi": 2,
        "arith.subi": 2,
        "arith.muli": 2,
        "arith.cmpi": 2,
        "arith.index_cast": 1,
        "arith.ceildivsi": 2,
        "arith.minui": 2,
        "arith.select": 3,
    }

    def num_regions(op: str) -> int:
        if op == "scf.for":
            return 1
        if op == "scf.if":
            return 2
        if op == "pto.section":
            return 1
        return 0

    def imm_kind(op: str) -> int:
        # 0=none
        # 1=cmpi_pred(u8)
        # 2=event3(u8,u8,u8)
        # 5=const_id(uLEB128)
        # 6=make_tensor_view: list_mode(u8), nshape(uLEB128), nstrides(uLEB128)
        # 7=partition_view: list_mode(u8), noffsets(uLEB128), nsizes(uLEB128)
        # 8=alloc_tile: opt_mask(u8) bit0=row bit1=col
        if op == "arith.cmpi":
            return 1
        if op in {"pto.record_event", "pto.wait_event"}:
            return 2
        if op == "arith.constant":
            return 5
        if op == "pto.make_tensor_view":
            return 6
        if op == "pto.partition_view":
            return 7
        if op == "pto.alloc_tile":
            return 8
        return 0

    def num_results(op: str) -> int:
        if op.startswith("arith."):
            return 1
        if op == "scf.yield":
            return 0
        if op in {
            "pto.get_block_idx",
            "pto.get_subblock_idx",
            "pto.get_block_num",
            "pto.get_subblock_num",
            "pto.addptr",
            "pto.make_tensor_view",
            "pto.partition_view",
            "pto.alloc_tile",
        }:
            return 1
        # Everything else is DPS style (0 results)
        return 0

    VARCOUNT_OPS = {
        # tsync may be variadic events
        "pto.tsync",
        # scf regions terminate with scf.yield, which may be variadic.
        "scf.yield",
    }

    BY_VARIANT_OPS = {k for k in fam_map.keys() if k.startswith("pto.tmatmul") or k.startswith("pto.tgemv")}

    def fixed_arity(op: str) -> int:
        if op in ARITH_ARITY:
            return ARITH_ARITY[op]
        if op == "scf.for":
            return 3
        if op == "scf.if":
            return 1
        if op == "func.return":
            return 0
        if op == "pto.section":
            return 0
        if op in {"pto.record_event", "pto.wait_event", "pto.barrier"}:
            return 0
        if op in {"pto.get_block_idx", "pto.get_subblock_idx", "pto.get_block_num", "pto.get_subblock_num"}:
            return 0
        if op == "pto.addptr":
            return 2
        # ISA + other PTO ops: look up DPS arity from docs
        if op in dps:
            return dps[op]
        # default: no operands
        return 0

    # Build schema + by-variant operand maps.
    schema_entries = {}
    by_variant_entries = {}  # opcode -> {variant_u8: operand_count}

    for r in rows:
        op = r.op
        opc = r.opcode

        has_variant = op in fam_map

        # result type ids: we choose explicit for all result-producing ops (lossless, no inference needed)
        rtm = 1 if num_results(op) > 0 else 0

        # operand_mode: 0=fixed, 1=by_variant, 2=varcount(total-prefix), 3=segmented(view), 4=optional_mask2
        if op in BY_VARIANT_OPS:
            operand_mode = 1
            nops = 0
            # Fill mapping for all variants (including base=0 when applicable)
            vmap = fam_map.get(op, {})
            m = {}
            for vlabel, vid in vmap.items():
                if vlabel == "base":
                    full = op
                else:
                    full = f"{op}.{vlabel}"

                # Prefer TableGen operand arity (authoritative for PTOAS IR ops).
                ar = tg_arity(full)
                if ar is None:
                    ar = dps.get(full, fixed_arity(full))
                m[vid] = ar

            by_variant_entries[opc] = m

        elif op in {"pto.make_tensor_view", "pto.partition_view"}:
            operand_mode = 3
            nops = 1  # base operand: ptr/source

        elif op == "pto.alloc_tile":
            operand_mode = 4
            nops = 0

        elif op in VARCOUNT_OPS:
            operand_mode = 2
            nops = 0

        else:
            operand_mode = 0
            nops = fixed_arity(op)

        sch = {
            "has_variant_u8": has_variant,
            "result_type_mode": rtm,
            "operand_mode": operand_mode,
            "num_operands": nops,
            "num_results": num_results(op),
            "num_regions": num_regions(op),
            "imm_kind": imm_kind(op),
        }
        schema_entries[opc] = sch

    # Write ISA DPS report.
    write_isa_report(rows, schema_entries, dps)

    # Emit ptobc C++ header.
    write_ptobc_header(rows, fam_map, schema_entries, by_variant_entries)

    # Emit Sail.
    lines = []
    lines.append("/* Generated: PTO-BC v0 opcode schema (full) */\n")
    lines.append("$ifndef PTO_BC_OPCODES_V0\n$define PTO_BC_OPCODES_V0\n\n")

    lines.append("/* Variant maps (label -> variant_u8) */\n")
    lines.append(f"/* {json.dumps(fam_map, sort_keys=True)} */\n\n")

    lines.append("val opcode_schema_builtin_gen : u16 -> option(opcode_schema)\n")
    lines.append("function opcode_schema_builtin_gen(op) =\n")
    lines.append("  match op {\n")
    for opc in sorted(schema_entries.keys()):
        sch = schema_entries[opc]
        hv = "true" if sch["has_variant_u8"] else "false"
        lines.append(
            "    0x%04X => Some({ has_variant_u8 = %s, result_type_mode = 0x%02X, operand_mode = 0x%02X, num_operands = %d, num_results = %d, num_regions = %d, imm_kind = 0x%02X })\n"
            % (
                opc,
                hv,
                sch["result_type_mode"],
                sch["operand_mode"],
                sch["num_operands"],
                sch["num_results"],
                sch["num_regions"],
                sch["imm_kind"],
            )
        )
    lines.append("  | _ => None()\n  }\n\n")

    lines.append("val opcode_operands_by_variant_gen : u16 -> int -> option(int)\n")
    lines.append("function opcode_operands_by_variant_gen(op, variant) =\n")
    lines.append("  match op {\n")
    for opc in sorted(by_variant_entries.keys()):
        vm = by_variant_entries[opc]
        lines.append(f"    0x{opc:04X} => match variant {{\n")
        for vid in sorted(vm.keys()):
            lines.append(f"      {vid} => Some({vm[vid]})\n")
        lines.append("    | _ => None()\n    }\n")
    lines.append("  | _ => None()\n  }\n\n")

    lines.append("$endif\n")

    OUT_SAIL.write_text("".join(lines), encoding="utf-8")


def _discover_matmul_gemv_families_from_tablegen():
    """Populate global FAMILIES/FAMILY_HAS_BASE_VARIANT for matmul/gemv from PTOOps.td."""
    global FAMILIES, FAMILY_HAS_BASE_VARIANT

    tg = parse_pto_ops_tablegen()
    fam: dict[str, set[str]] = {k: set(vs) for k, vs in FAMILIES.items()}
    has_base: dict[str, bool] = dict(FAMILY_HAS_BASE_VARIANT)

    def add_variant(base: str, variant: str | None):
        fam.setdefault(base, set())
        if variant is None:
            has_base[base] = True
            fam[base].add("base")
        else:
            has_base[base] = True
            fam[base].add(variant)

    for op in tg.keys():
        if not (op.startswith("pto.tmatmul") or op.startswith("pto.tgemv")):
            continue

        # Handle nested `.mx` base if present.
        if op.startswith("pto.tmatmul.mx"):
            base = "pto.tmatmul.mx"
            rest = op[len(base):]
        elif op.startswith("pto.tgemv.mx"):
            base = "pto.tgemv.mx"
            rest = op[len(base):]
        elif op.startswith("pto.tmatmul"):
            base = "pto.tmatmul"
            rest = op[len(base):]
        else:
            base = "pto.tgemv"
            rest = op[len(base):]

        if rest == "":
            add_variant(base, None)
        elif rest.startswith("."):
            add_variant(base, rest[1:])

    # Write back discovered families (keep section family).
    # Convert sets to sorted lists excluding implicit 'base'.
    new_families = {"pto.section": ["cube", "vector"]}
    new_has_base = {"pto.section": False}

    for base, vs in fam.items():
        if base == "pto.section":
            continue
        # Keep only if we actually discovered something beyond empty.
        if not vs:
            continue
        # Store without 'base' label; `assign_opcodes` will add it back based on FAMILY_HAS_BASE_VARIANT.
        new_families[base] = sorted(v for v in vs if v != "base")
        new_has_base[base] = True

    FAMILIES = new_families
    FAMILY_HAS_BASE_VARIANT = new_has_base


def main():
    # Discover matmul/gemv family variants from PTOAS TableGen.
    _discover_matmul_gemv_families_from_tablegen()

    isa = load_isa_ops()
    ir = parse_ir_doc_ops()
    smp = parse_sample_ops()

    all_ops = set().union(isa, ir, smp)
    # MLIR structured control flow terminator (required inside scf regions).
    all_ops.add("scf.yield")

    rows, fam_map = assign_opcodes(all_ops)

    write_md(rows, fam_map)
    write_families_json(fam_map)
    write_sail(rows, fam_map)

    print(f"Wrote: {OUT_MD}")
    print(f"Wrote: {OUT_FAMILIES}")
    print(f"Wrote: {OUT_SAIL}")


if __name__ == "__main__":
    main()
