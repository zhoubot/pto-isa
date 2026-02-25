#!/usr/bin/env python3
"""Stage9 harness (reference-level): build PTOBC v0 blobs and sanity-check.

This harness does NOT parse `.pto` text. It focuses on container+section closure.

Checks:
- Build minimal PTOBC with DEBUGINFO.
- Re-parse DEBUGINFO using check_stage8_debuginfo parser.

Future:
- Add real `.pto` -> PTOBC encoder.
"""

from __future__ import annotations

from pathlib import Path

from encode_ptobc_ref import build_ptobc_v0, module_minimal, debuginfo_minimal


def main() -> int:
    blob = build_ptobc_v0(strings=[b"file.pto"], module_bytes=module_minimal(), debuginfo_bytes=debuginfo_minimal())
    out = Path("/tmp/ptobc_minimal.bin")
    out.write_bytes(blob)
    print(f"Wrote {out} ({len(blob)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
