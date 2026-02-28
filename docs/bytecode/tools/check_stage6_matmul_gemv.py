#!/usr/bin/env python3
"""Stage6 closure check: matmul/gemv families compact schema.

Validates that for each family base op:
- opcode schema uses operand_mode=by_variant
- opcode_operands_by_variant_gen contains operand count per variant
- counts match PTOAS TableGen (PTOOps.td) for the corresponding op spelling

Does NOT require Sail to be installed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPCODES_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
SCHEMA_SAIL = ROOT / "sail/generated/pto_bc_opcodes_v0.sail"
FAM_JSON = ROOT / "docs/bytecode/generated/op_families_v0.json"
PTOOPS_TD = ROOT / "PTOAS/include/PTO/IR/PTOOps.td"


def load_opcode_map() -> dict[str, int]:
    op_to_opcode = {}
    for ln in OPCODES_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- `0x([0-9A-Fa-f]{4})`\s+[A-Za-z0-9_-]+\s+`([^`]+)`", ln)
        if m:
            op_to_opcode[m.group(2)] = int(m.group(1), 16)
    return op_to_opcode


def load_schema_map() -> dict[int, dict[str, int]]:
    schema = {}
    pat = re.compile(
        r"\s*0x([0-9A-Fa-f]{4}) => Some\(\{.*?result_type_mode = 0x([0-9A-Fa-f]{2}), operand_mode = 0x([0-9A-Fa-f]{2}), num_operands = (\d+), num_results = (\d+), num_regions = (\d+), imm_kind = 0x([0-9A-Fa-f]{2})",
        re.S,
    )
    for ln in SCHEMA_SAIL.read_text(encoding="utf-8").splitlines():
        m = pat.match(ln)
        if m:
            opc = int(m.group(1), 16)
            schema[opc] = {
                "result_type_mode": int(m.group(2), 16),
                "operand_mode": int(m.group(3), 16),
                "num_operands": int(m.group(4)),
                "num_results": int(m.group(5)),
                "num_regions": int(m.group(6)),
                "imm_kind": int(m.group(7), 16),
            }
    return schema


def load_by_variant_map() -> dict[int, dict[int, int]]:
    """Parse opcode_operands_by_variant_gen from generated Sail."""
    text = SCHEMA_SAIL.read_text(encoding="utf-8")
    out: dict[int, dict[int, int]] = {}

    # Match blocks like:
    # 0x1033 => match variant {
    #   0 => Some(4)
    # | _ => None()
    # }
    block_pat = re.compile(r"\s*0x([0-9A-Fa-f]{4}) => match variant \{(?P<body>.*?)\n\s*\| _ => None\(\)\n\s*\}\n", re.S)
    ent_pat = re.compile(r"\s*([0-9]+) => Some\(([0-9]+)\)")

    for m in block_pat.finditer(text):
        opc = int(m.group(1), 16)
        body = m.group("body")
        mp: dict[int, int] = {}
        for em in ent_pat.finditer(body):
            mp[int(em.group(1))] = int(em.group(2))
        out[opc] = mp

    return out


def parse_pto_ops_tablegen() -> dict[str, int]:
    """Map pto.<opname> -> min_operands for fixed-arity ops (no optional/variadic)."""
    txt = PTOOPS_TD.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=def\s+)", txt)

    op_pat = re.compile(r"def\s+\w+\s*:\s*PTO_[A-Za-z0-9_]*Op<\"([^\"]+)\"")
    args_pat = re.compile(r"let\s+arguments\s*=\s*\(ins(?P<body>.*?)\);", re.S)
    entry_pat = re.compile(r"(?P<ty>[^:()\n]+):\$(?P<name>[A-Za-z0-9_]+)")

    def is_attr_type(ty: str) -> bool:
        ty = ty.strip()
        if "OptionalAttr<" in ty:
            return True
        if re.search(r"\b[A-Za-z0-9_]*Attr\b", ty):
            return True
        if "Attr<" in ty:
            return True
        return False

    out: dict[str, int] = {}

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

        min_ops = 0
        has_var = False
        has_opt = False

        for em in entry_pat.finditer(body):
            ty = em.group("ty").strip()
            if is_attr_type(ty):
                continue
            is_var = "Variadic<" in ty
            is_opt = "Optional<" in ty
            if is_var:
                has_var = True
            if is_opt:
                has_opt = True
            if (not is_var) and (not is_opt):
                min_ops += 1

        if not has_var and not has_opt:
            out[op_full] = min_ops

    return out


def main() -> int:
    import json

    fam = json.loads(FAM_JSON.read_text(encoding="utf-8"))
    op_to_opcode = load_opcode_map()
    schema = load_schema_map()
    byv = load_by_variant_map()
    tg = parse_pto_ops_tablegen()

    families = [k for k in fam.keys() if k.startswith("pto.tmatmul") or k.startswith("pto.tgemv")]

    ok = True
    for base in sorted(families):
        if base not in op_to_opcode:
            print(f"[FAIL] missing opcode for family base {base}")
            ok = False
            continue
        opc = op_to_opcode[base]
        sch = schema.get(opc)
        if not sch:
            print(f"[FAIL] missing schema for {base} (0x{opc:04X})")
            ok = False
            continue
        if sch["operand_mode"] != 0x01:
            print(f"[FAIL] {base} operand_mode not by_variant: got=0x{sch['operand_mode']:02X}")
            ok = False

        vmap = fam[base]
        opmap = byv.get(opc)
        if not opmap:
            print(f"[FAIL] missing by-variant table for {base} (0x{opc:04X})")
            ok = False
            continue

        for vlabel, vid in vmap.items():
            # Map to full op name
            if vlabel == "base":
                full = base
            else:
                full = f"{base}.{vlabel}"

            # Compare operand counts
            exp = tg.get(full)
            if exp is None:
                print(f"[FAIL] PTOOps.td has no fixed-arity entry for {full}")
                ok = False
                continue

            got = opmap.get(int(vid))
            if got is None:
                print(f"[FAIL] missing operand count for {base} variant {vlabel} (id={vid})")
                ok = False
                continue
            if int(got) != int(exp):
                print(f"[FAIL] operand count mismatch {full}: got={got} exp={exp}")
                ok = False

    if ok:
        print("[PASS] Stage6 matmul/gemv by-variant compact schema check")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
