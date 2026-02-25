#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def extract_instr_names(pto_instr_hpp: str) -> list[str]:
    names: set[str] = set()
    for line in pto_instr_hpp.splitlines():
        if "PTO_INST" not in line:
            continue
        m = re.search(r"\bPTO_INST\b.*?\b([A-Z][A-Z0-9_]*)\s*\(", line)
        if m:
            names.add(m.group(1))
    return sorted(names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate spec-only PTO ODS op stubs from include/pto/common/pto_instr.hpp.")
    ap.add_argument("--input", type=Path, default=repo_root() / "include/pto/common/pto_instr.hpp")
    ap.add_argument("--output", type=Path, default=repo_root() / "ptoas/PTOASOps.td")
    ap.add_argument("--dialect-op-prefix", default="PTO_", help="ODS def prefix (default: PTO_)")
    ap.add_argument(
        "--skip",
        action="append",
        default=["tadd", "tload", "tstore", "mgather", "mscatter", "tmrgsort"],
        help="Skip mnemonics that are already hand-written in PTOAS.td (repeatable).",
    )
    args = ap.parse_args()

    text = args.input.read_text(encoding="utf-8", errors="ignore")
    names = extract_instr_names(text)
    skip = set(args.skip)

    lines: list[str] = []
    lines.append("//===- PTOASOps.td --------------------------------------------*- tablegen -*-===//")
    lines.append("//")
    lines.append("// Auto-generated spec-only op stubs for PTO-AS.")
    lines.append("//")
    lines.append("// Regenerate with:")
    lines.append("//   python3 ptoas/tools/gen_ptoas_ops_td.py")
    lines.append("//")
    lines.append("// Source of truth for instruction names:")
    lines.append("//   include/pto/common/pto_instr.hpp")
    lines.append("//")
    lines.append("// NOTE: These are intentionally generic (variadic operands).")
    lines.append("//       A full dialect would model operands/results/types precisely.")
    lines.append("//")
    lines.append("//===----------------------------------------------------------------------===//")
    lines.append("")

    for api in names:
        mnemonic = api.lower()
        if mnemonic in skip:
            continue
        ods_name = f"{args.dialect_op_prefix}{api.title().replace('_', '')}Op"
        lines.append(f"def {ods_name} : PTO_Op<\"{mnemonic}\", [MemoryEffects<[MemWrite, MemRead]>]> {{")
        lines.append(f"  let summary = \"{api} (auto-generated stub)\";")
        lines.append("  let arguments = (ins Variadic<AnyType>:$operands);")
        lines.append("  let assemblyFormat = \"$operands attr-dict\";")
        lines.append("}")
        lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

