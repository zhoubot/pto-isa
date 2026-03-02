#!/usr/bin/env python3
"""Reference PTO-BC v0 encoder/decoder helpers (Python).

This is a *developer tool* to support Stage9 integration tests.

- Encoder path may depend on MLIR Python bindings (to parse `.pto`).
- Decoder/validator should be MLIR-independent in the long run; for now we provide
  a reference decoder that can reconstruct an MLIR module (generic op form) when
  MLIR bindings are available.

NOTE: This file is not production-grade and aims to cover the PTODSL sample set.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[3]

OPCODES_MD = ROOT / "docs/bytecode/generated/opcodes_v0.md"
FAMILIES_JSON = ROOT / "docs/bytecode/generated/op_families_v0.json"
SCHEMA_SAIL = ROOT / "tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail"

MAGIC = b"PTOBC\x00"

# ----------------------------
# basic encodings


def u16le(x: int) -> bytes:
    return struct.pack("<H", x)


def u32le(x: int) -> bytes:
    return struct.pack("<I", x)


def uleb(x: int) -> bytes:
    assert x >= 0
    out = bytearray()
    while True:
        b = x & 0x7F
        x >>= 7
        if x:
            out.append(b | 0x80)
        else:
            out.append(b)
            break
    return bytes(out)


def sleb(x: int) -> bytes:
    out = bytearray()
    more = True
    while more:
        b = x & 0x7F
        x_shifted = x >> 7
        sign_bit = b & 0x40
        more = not ((x_shifted == 0 and sign_bit == 0) or (x_shifted == -1 and sign_bit != 0))
        if more:
            b |= 0x80
        out.append(b & 0xFF)
        x = x_shifted
    return bytes(out)


def sec(sec_id: int, data: bytes) -> bytes:
    return struct.pack("<B", sec_id) + u32le(len(data)) + data


# ----------------------------
# tables loaded from repo


@dataclass
class Schema:
    has_variant_u8: bool
    result_type_mode: int
    operand_mode: int
    num_operands: int
    num_results: int
    num_regions: int
    imm_kind: int


def load_opcode_map() -> Tuple[Dict[str, int], Dict[int, str]]:
    op_to_opcode: Dict[str, int] = {}
    opcode_to_op: Dict[int, str] = {}
    for ln in OPCODES_MD.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^- `0x([0-9A-Fa-f]{4})`\s+[A-Za-z0-9_-]+\s+`([^`]+)`", ln)
        if not m:
            continue
        opc = int(m.group(1), 16)
        op = m.group(2)
        op_to_opcode[op] = opc
        opcode_to_op[opc] = op
    return op_to_opcode, opcode_to_op


def load_families() -> Dict[str, Dict[str, int]]:
    return json.loads(FAMILIES_JSON.read_text(encoding="utf-8"))


def load_schema_map() -> Dict[int, Schema]:
    out: Dict[int, Schema] = {}
    pat = re.compile(
        r"\s*0x([0-9A-Fa-f]{4}) => Some\(\{\s*has_variant_u8 = (true|false),\s*result_type_mode = 0x([0-9A-Fa-f]{2}),\s*operand_mode = 0x([0-9A-Fa-f]{2}),\s*num_operands = (\d+),\s*num_results = (\d+),\s*num_regions = (\d+),\s*imm_kind = 0x([0-9A-Fa-f]{2})\s*\}\)",
        re.S,
    )
    for ln in SCHEMA_SAIL.read_text(encoding="utf-8").splitlines():
        m = pat.match(ln)
        if not m:
            continue
        opc = int(m.group(1), 16)
        out[opc] = Schema(
            has_variant_u8=(m.group(2) == "true"),
            result_type_mode=int(m.group(3), 16),
            operand_mode=int(m.group(4), 16),
            num_operands=int(m.group(5)),
            num_results=int(m.group(6)),
            num_regions=int(m.group(7)),
            imm_kind=int(m.group(8), 16),
        )
    return out


# ----------------------------
# string/type/attr interns


class Intern:
    def __init__(self):
        self.strings: List[bytes] = []
        self.s_map: Dict[bytes, int] = {}

    def sid(self, s: str) -> int:
        b = s.encode("utf-8")
        return self.sid_bytes(b)

    def sid_bytes(self, b: bytes) -> int:
        if b in self.s_map:
            return self.s_map[b]
        i = len(self.strings)
        self.strings.append(b)
        self.s_map[b] = i
        return i

    def emit_strings(self) -> bytes:
        out = bytearray()
        out += uleb(len(self.strings))
        for b in self.strings:
            out += uleb(len(b))
            out += b
        return bytes(out)


@dataclass
class TypeEntry:
    tag: int
    asm_sid: int
    payload: bytes


class TypeTable:
    def __init__(self, I: Intern):
        self.I = I
        self.entries: List[TypeEntry] = []
        self.map: Dict[str, int] = {}

    def type_id(self, asm: str) -> int:
        if asm in self.map:
            return self.map[asm]
        sid = self.I.sid(asm)
        # Opaque type with asm backup.
        ent = TypeEntry(tag=0x00, asm_sid=sid, payload=b"")
        tid = len(self.entries)
        self.entries.append(ent)
        self.map[asm] = tid
        return tid

    def func_type_id(self, arg_type_ids: List[int], res_type_ids: List[int], asm: str) -> int:
        if asm in self.map:
            return self.map[asm]
        sid = self.I.sid(asm)
        payload = bytearray()
        payload += uleb(len(arg_type_ids))
        for t in arg_type_ids:
            payload += uleb(t)
        payload += uleb(len(res_type_ids))
        for t in res_type_ids:
            payload += uleb(t)
        ent = TypeEntry(tag=0x20, asm_sid=sid, payload=bytes(payload))
        tid = len(self.entries)
        self.entries.append(ent)
        self.map[asm] = tid
        return tid

    def emit(self) -> bytes:
        out = bytearray()
        out += uleb(len(self.entries))
        for e in self.entries:
            out.append(e.tag & 0xFF)
            out.append(0x01)  # flags: has_asm
            out += uleb(e.asm_sid)
            out += e.payload
        return bytes(out)


class AttrTable:
    def __init__(self, I: Intern):
        self.I = I
        self.entries: List[int] = []  # asm_sid
        self.map: Dict[str, int] = {}

    def attr_id(self, asm: str) -> int:
        if asm == "" or asm == "{}":
            return 0
        if asm in self.map:
            return self.map[asm]
        sid = self.I.sid(asm)
        idx = len(self.entries)
        self.entries.append(sid)
        # attr_id is 1-based (0 means none)
        aid = idx + 1
        self.map[asm] = aid
        return aid

    def emit(self) -> bytes:
        out = bytearray()
        out += uleb(len(self.entries))
        for sid in self.entries:
            out.append(0x00)  # tag
            out.append(0x01)  # flags: has_asm
            out += uleb(sid)
        return bytes(out)


# ----------------------------
# minimal constpool (ints only for now)


@dataclass(frozen=True)
class ConstKey:
    type_id: int
    value: int


class ConstPool:
    def __init__(self):
        self.entries: List[ConstKey] = []
        self.map: Dict[ConstKey, int] = {}

    def const_int(self, type_id: int, value: int) -> int:
        k = ConstKey(type_id, value)
        if k in self.map:
            return self.map[k]
        cid = len(self.entries)
        self.entries.append(k)
        self.map[k] = cid
        return cid

    def emit(self) -> bytes:
        out = bytearray()
        out += uleb(len(self.entries))
        for k in self.entries:
            out.append(0x01)  # int tag
            out += uleb(k.type_id)
            out += sleb(k.value)
        return bytes(out)
