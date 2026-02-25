#!/usr/bin/env python3
"""Stage5 closure check: ISA (docs/isa DPS) ops compact schema correctness.

Checks that all PTO ISA ops spelled in docs/isa as DPS lines:
  pto.xxx ins(...) outs(...)
are present in the frozen opcode table and have compact schemas.

Excludes matmul/gemv families (Stage6) and tsync/event (Stage7).

Does NOT require Sail to be installed.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ISA_DIR = ROOT / "docs/isa"
OPCODES_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
SCHEMA_SAIL = ROOT / "sail/generated/pto_bc_opcodes_v0.sail"

SKIP_PREFIX = (
    "pto.tmatmul",
    "pto.tgemv",
)


def count_args(seg: str) -> int:
    seg = seg.strip()
    if not seg:
        return 0
    seg = seg.split(":", 1)[0].strip().rstrip(")")
    if seg.strip() == "":
        return 0
    return sum(1 for t in (x.strip() for x in seg.split(",")) if t)


def parse_dps_arity() -> dict[str, int]:
    out: dict[str, int] = {}
    for md in sorted(ISA_DIR.glob("*.md")):
        txt = md.read_text(encoding="utf-8")
        for line in txt.splitlines():
            if "pto." not in line or "ins(" not in line or "outs(" not in line:
                continue
            m = re.search(r"\b(pto\.[A-Za-z0-9_.]+)\b", line)
            if not m:
                continue
            op = m.group(1)
            mi = re.search(r"ins\(([^)]*)\)", line)
            mo = re.search(r"outs\(([^)]*)\)", line)
            if not (mi and mo):
                continue
            ar = count_args(mi.group(1)) + count_args(mo.group(1))
            out[op] = max(out.get(op, 0), ar)
    return out


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


def main() -> int:
    dps = parse_dps_arity()
    op_to_opcode = load_opcode_map()
    schema = load_schema_map()

    ok = True

    for op, arity in sorted(dps.items()):
        if op.startswith(SKIP_PREFIX):
            continue
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

        # Stage5 expects fixed operand count from DPS.
        if got["operand_mode"] != 0x00:
            print(f"[FAIL] {op} operand_mode not fixed: got=0x{got['operand_mode']:02X}")
            ok = False
        if got["num_operands"] != arity:
            print(f"[FAIL] {op} num_operands mismatch: got={got['num_operands']} exp={arity}")
            ok = False
        if got["num_regions"] != 0:
            print(f"[FAIL] {op} should have no regions: got={got['num_regions']}")
            ok = False
        if got["imm_kind"] != 0x00:
            print(f"[FAIL] {op} should have no imm_kind in Stage5: got=0x{got['imm_kind']:02X}")
            ok = False

    if ok:
        print("[PASS] Stage5 ISA DPS compact schema check")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
