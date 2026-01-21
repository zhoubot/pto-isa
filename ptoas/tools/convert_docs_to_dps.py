#!/usr/bin/env python3
# coding=utf-8

import re
from pathlib import Path
from typing import Optional


ASSIGN_RE = re.compile(
    r"^(?P<lhs>\s*%[\w\d_]+(?:\s*,\s*%[\w\d_]+)*)\s*=\s*(?P<op>t[\w\.\d_]+)\s*(?P<rhs>.*?)\s*:\s*(?P<types>.+?)\s*$"
)


def _split_top_level_csv(s: str, extra_pairs: str = "[]()<>") -> list[str]:
    pairs = {"[": "]", "(": ")", "<": ">"}
    close_to_open = {v: k for k, v in pairs.items()}
    opens = set(pairs.keys())
    closes = set(pairs.values())

    stack: list[str] = []
    parts: list[str] = []
    buf: list[str] = []

    for ch in s:
        if ch in opens and ch in extra_pairs:
            stack.append(ch)
        elif ch in closes and ch in extra_pairs:
            if stack and stack[-1] == close_to_open.get(ch):
                stack.pop()
        if ch == "," and not stack:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
            continue
        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def _strip_stmt_suffix(s: str) -> str:
    s = s.strip()
    if s.endswith(";"):
        return s[:-1].rstrip()
    return s


def rewrite_line_to_dps(line: str) -> Optional[str]:
    raw = line.strip()
    m = ASSIGN_RE.match(raw)
    if not m:
        return None

    lhs_vals = [x.strip() for x in _split_top_level_csv(m.group("lhs"))]
    op = m.group("op").strip()
    rhs = _strip_stmt_suffix(m.group("rhs"))
    types = _strip_stmt_suffix(m.group("types"))

    rhs_vals = []
    if rhs:
        rhs_vals = [x.strip() for x in _split_top_level_csv(rhs)]

    # Rebuild operands: default outputs first, then original rhs operands.
    operands = lhs_vals + rhs_vals
    if op == "tsort32" and len(lhs_vals) == 2 and len(rhs_vals) == 1:
        # C++ signature is TSORT32(dst, src, idx)
        operands = [lhs_vals[0], rhs_vals[0], lhs_vals[1]]

    # Rebuild types:
    # - If old type contains "->", move result types before input types.
    # - Otherwise treat as comma-list of operand types for RHS and replicate for LHS.
    if "->" in types:
        in_part, out_part = [x.strip() for x in types.split("->", 1)]
        if in_part.startswith("(") and in_part.endswith(")"):
            in_types = _split_top_level_csv(in_part[1:-1])
        else:
            in_types = _split_top_level_csv(in_part)

        if out_part.startswith("(") and out_part.endswith(")"):
            out_types = _split_top_level_csv(out_part[1:-1])
        else:
            out_types = [_strip_stmt_suffix(out_part)]

        all_types = out_types + in_types
        new_types = "(" + ", ".join(all_types) + ")"
    else:
        rhs_types = _split_top_level_csv(types)
        if not rhs_types:
            return None
        rhs_operand_count = len(rhs_vals)
        if len(rhs_types) == 1 and rhs_operand_count > 1:
            rhs_types = rhs_types * rhs_operand_count
        elif len(rhs_types) < rhs_operand_count and rhs_operand_count > 0:
            rhs_types = rhs_types + [rhs_types[-1]] * (rhs_operand_count - len(rhs_types))

        # Replicate the first RHS type for each LHS value (dst tile type).
        lhs_types = [rhs_types[0]] * len(lhs_vals) if rhs_types else []
        all_types = lhs_types + rhs_types
        new_types = "(" + ", ".join(all_types) + ")"

    return f"{op} {', '.join(operands)} : {new_types}"


def rewrite_md_file(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=False)

    in_fence = False
    fence_lang = None
    changed = False
    out_lines: list[str] = []

    for line in lines:
        if line.strip().startswith("```"):
            if not in_fence:
                in_fence = True
                fence_lang = line.strip()[3:].strip()
            else:
                in_fence = False
                fence_lang = None
            out_lines.append(line)
            continue

        if in_fence and (fence_lang in ("text", "asm")):
            repl = rewrite_line_to_dps(line)
            if repl is not None:
                out_lines.append(repl)
                changed = True
                continue

        out_lines.append(line)

    if changed:
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return changed


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    targets = list((repo_root / "docs" / "isa").glob("*.md"))

    changed_files = 0
    for p in sorted(targets):
        if p.exists() and rewrite_md_file(p):
            changed_files += 1

    print(f"updated_files={changed_files}")


if __name__ == "__main__":
    main()
