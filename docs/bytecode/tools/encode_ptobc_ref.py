#!/usr/bin/env python3
"""Reference PTO-BC v0 encoder (minimal) for Stage9 harness.

This is NOT a full `.pto` parser.

It builds a PTOBC file from already-structured inputs (strings/types/attrs/consts/module).
Stage9 uses it to test container layout + tables + module framing + debuginfo.

A full `.pto` -> PTOBC encoder will be a separate component.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

MAGIC = b"PTOBC\x00"


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
        x_shift = x >> 7
        sign = b & 0x40
        more = not ((x_shift == 0 and sign == 0) or (x_shift == -1 and sign != 0))
        if more:
            b |= 0x80
        out.append(b & 0xFF)
        x = x_shift
    return bytes(out)


def u16le(x: int) -> bytes:
    return struct.pack("<H", x)


def u32le(x: int) -> bytes:
    return struct.pack("<I", x)


def sec(sec_id: int, data: bytes) -> bytes:
    return struct.pack("<B", sec_id) + u32le(len(data)) + data


def strings_table(strs: list[bytes]) -> bytes:
    out = bytearray()
    out += uleb(len(strs))
    for s in strs:
        out += uleb(len(s))
        out += s
    return bytes(out)


def types_table_opaque(count: int) -> bytes:
    # placeholder: 0 entries
    return uleb(count)


def attrs_table_empty() -> bytes:
    return uleb(0)


def constpool_empty() -> bytes:
    return uleb(0)


def module_minimal() -> bytes:
    # profile_id=0, index_width=64
    # module_attr_id=0
    # globals=0
    # funcs=0
    return bytes([0, 64]) + uleb(0) + uleb(0) + uleb(0)


def debuginfo_minimal() -> bytes:
    out = bytearray()
    out += uleb(0)  # files
    out += uleb(0)  # value names
    out += uleb(0)  # locations
    out += uleb(0)  # snippets
    return bytes(out)


def build_ptobc_v0(*, strings: list[bytes], module_bytes: bytes, debuginfo_bytes: bytes | None = None) -> bytes:
    payload_parts = [
        sec(0x01, strings_table(strings)),
        sec(0x02, types_table_opaque(0)),
        sec(0x03, attrs_table_empty()),
        sec(0x04, constpool_empty()),
        sec(0x06, module_bytes),
    ]
    if debuginfo_bytes is not None:
        payload_parts.append(sec(0x07, debuginfo_bytes))
    payload = b"".join(payload_parts)

    header = MAGIC + u16le(0) + u16le(0) + u32le(len(payload))
    return header + payload
