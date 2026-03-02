#!/usr/bin/env python3
"""Stage7 closure check: events / barrier / tsync compact schema.

- pto.record_event / pto.wait_event: imm_kind=event3 (0x02), fixed operands=0
- pto.barrier: fixed operands=0
- pto.tsync: operand_mode=varcount (0x02)

Also checks that the Stage7 sample exists.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OPCODES_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
SCHEMA_SAIL = ROOT / "tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail"
SAMPLE = ROOT / "docs/bytecode/samples/sync_stage7.pto"

EXPECT = {
    "pto.record_event": {"operand_mode": 0x00, "num_operands": 0, "num_results": 0, "num_regions": 0, "imm_kind": 0x02},
    "pto.wait_event": {"operand_mode": 0x00, "num_operands": 0, "num_results": 0, "num_regions": 0, "imm_kind": 0x02},
    "pto.barrier": {"operand_mode": 0x00, "num_operands": 0, "num_results": 0, "num_regions": 0, "imm_kind": 0x00},
    "pto.tsync": {"operand_mode": 0x02, "num_operands": 0, "num_results": 0, "num_regions": 0, "imm_kind": 0x00},
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


def main() -> int:
    if not SAMPLE.exists():
        print(f"[FAIL] missing sample: {SAMPLE}")
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
        print("[PASS] Stage7 sync/event compact schema check")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
