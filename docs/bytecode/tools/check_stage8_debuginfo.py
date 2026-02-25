#!/usr/bin/env python3
"""Stage8 closure check (partial): DEBUGINFO section format.

Builds a minimal PTOBC v0 file (with empty module) that includes a DEBUGINFO section,
then parses it back and checks round-trip of key fields.

This is a reference check independent of Sail installation.
"""

from __future__ import annotations

import struct

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


def empty_table() -> bytes:
    return uleb(0)


def module_minimal() -> bytes:
    # profile_id=0, index_width=64
    # module_attr_id=0
    # globals=0
    # funcs=0
    return bytes([0, 64]) + uleb(0) + uleb(0) + uleb(0)


def debuginfo_minimal() -> bytes:
    # file table: 1 entry
    # value names: 1
    # locations: 1
    # snippets: 1
    out = bytearray()

    # files
    out += uleb(1)
    out += uleb(0)  # path_sid
    out += uleb(0)  # hash_len

    # value names
    out += uleb(1)
    out += uleb(0)  # func_id
    out += uleb(0)  # value_id
    out += uleb(1)  # name_sid

    # locations
    out += uleb(1)
    out += uleb(0)  # func_id
    out += uleb(0)  # op_id
    out += uleb(0)  # file_id
    out += uleb(1)  # sl
    out += uleb(1)  # sc
    out += uleb(1)  # el
    out += uleb(2)  # ec

    # snippets
    out += uleb(1)
    out += uleb(0)  # func_id
    out += uleb(0)  # op_id
    out += uleb(2)  # snippet_sid

    return bytes(out)


def parse_uleb(buf: bytes, off: int) -> tuple[int, int]:
    shift = 0
    acc = 0
    while True:
        b = buf[off]
        off += 1
        acc |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return acc, off
        shift += 7


def parse_strings(buf: bytes) -> list[bytes]:
    off = 0
    cnt, off = parse_uleb(buf, off)
    out = []
    for _ in range(cnt):
        ln, off = parse_uleb(buf, off)
        out.append(buf[off : off + ln])
        off += ln
    assert off == len(buf)
    return out


def parse_debuginfo(buf: bytes):
    off = 0

    fcnt, off = parse_uleb(buf, off)
    files = []
    for _ in range(fcnt):
        path_sid, off = parse_uleb(buf, off)
        hlen, off = parse_uleb(buf, off)
        hb = buf[off : off + hlen]
        off += hlen
        files.append((path_sid, hb))

    vcnt, off = parse_uleb(buf, off)
    vnames = []
    for _ in range(vcnt):
        fn, off = parse_uleb(buf, off)
        vid, off = parse_uleb(buf, off)
        nsid, off = parse_uleb(buf, off)
        vnames.append((fn, vid, nsid))

    lcnt, off = parse_uleb(buf, off)
    locs = []
    for _ in range(lcnt):
        fn, off = parse_uleb(buf, off)
        opid, off = parse_uleb(buf, off)
        fid, off = parse_uleb(buf, off)
        sl, off = parse_uleb(buf, off)
        sc, off = parse_uleb(buf, off)
        el, off = parse_uleb(buf, off)
        ec, off = parse_uleb(buf, off)
        locs.append((fn, opid, fid, sl, sc, el, ec))

    scnt, off = parse_uleb(buf, off)
    snips = []
    for _ in range(scnt):
        fn, off = parse_uleb(buf, off)
        opid, off = parse_uleb(buf, off)
        ssid, off = parse_uleb(buf, off)
        snips.append((fn, opid, ssid))

    assert off == len(buf)
    return files, vnames, locs, snips


def main() -> int:
    strs = [b"sync_stage7.pto", b"%v0", b"pto.barrier"]

    payload = b"".join(
        [
            sec(0x01, strings_table(strs)),
            sec(0x02, empty_table()),
            sec(0x03, empty_table()),
            sec(0x04, empty_table()),
            sec(0x06, module_minimal()),
            sec(0x07, debuginfo_minimal()),
        ]
    )

    header = MAGIC + u16le(0) + u16le(0) + u32le(len(payload))
    blob = header + payload

    # Parse back
    assert blob[:6] == MAGIC
    payload_len = struct.unpack("<I", blob[10:14])[0]
    assert payload_len == len(payload)

    # Parse sections
    off = 14
    sections = {}
    while off < len(blob):
        sec_id = blob[off]
        slen = struct.unpack("<I", blob[off + 1 : off + 5])[0]
        off += 5
        data = blob[off : off + slen]
        off += slen
        sections[sec_id] = data

    s_strs = parse_strings(sections[0x01])
    files, vnames, locs, snips = parse_debuginfo(sections[0x07])

    assert s_strs[0] == b"sync_stage7.pto"
    assert files == [(0, b"")]
    assert vnames == [(0, 0, 1)]
    assert locs == [(0, 0, 0, 1, 1, 1, 2)]
    assert snips == [(0, 0, 2)]

    print("[PASS] Stage8 DEBUGINFO minimal format check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
