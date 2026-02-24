#!/usr/bin/env python3
"""Stage4 closure check: scf.* compact schema correctness.

Verifies that all scf ops used by PTODSL `.pto` samples:
- exist in the generated opcode table
- have a generated compact schema entry
- have expected operand counts and region counts

Does NOT require Sail to be installed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPCODES_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
SCHEMA_SAIL = ROOT / "sail/generated/pto_bc_opcodes_v0.sail"
SAMPLES = ROOT / "docs/bytecode/samples"

EXPECT = {
    "scf.for": {"num_operands": 3, "num_results": 0, "num_regions": 1, "imm_kind": 0x00, "operand_mode": 0x00, "result_type_mode": 0x00},
    "scf.if": {"num_operands": 1, "num_results": 0, "num_regions": 2, "imm_kind": 0x00, "operand_mode": 0x00, "result_type_mode": 0x00},
    # Terminator inside scf regions; operand count may vary.
    "scf.yield": {"num_operands": 0, "num_results": 0, "num_regions": 0, "imm_kind": 0x00, "operand_mode": 0x02, "result_type_mode": 0x00},
}


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


def find_scf_ops_in_samples() -> set[str]:
    ops = set()
    op_pat = re.compile(r"\bscf\.[A-Za-z0-9_]+\b")
    for p in sorted(SAMPLES.glob("*.pto")):
        txt = p.read_text(encoding="utf-8")
        ops.update(op_pat.findall(txt))
    return ops


def main() -> int:
    used = sorted(find_scf_ops_in_samples())
    unknown_used = [x for x in used if x not in EXPECT]
    if unknown_used:
        print("[FAIL] scf ops used in samples but not expected:")
        for x in unknown_used:
            print("  -", x)
        return 1

    op_to_opcode = load_opcode_map()
    schema = load_schema_map()

    ok = True
    for op, exp in EXPECT.items():
        if op not in op_to_opcode:
            print(f"[FAIL] missing opcode assignment for {op}")
            ok = False
            continue
        opc = op_to_opcode[op]
        if opc not in schema:
            print(f"[FAIL] missing schema entry for {op} (0x{opc:04X})")
            ok = False
            continue
        got = schema[opc]
        for k, v in exp.items():
            if got.get(k) != v:
                print(f"[FAIL] {op} schema mismatch {k}: got={got.get(k)} exp={v}")
                ok = False

    if ok:
        print("[PASS] Stage4 scf compact schema check")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
