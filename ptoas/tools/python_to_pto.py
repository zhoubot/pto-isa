#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ptoas.python import binding  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Python kernel -> PTO-AS (.pto) with embedded host metadata.")
    ap.add_argument("py", type=Path, help="Python kernel file")
    ap.add_argument("--kernel", help="Function name to compile (required if file has multiple defs)")
    ap.add_argument("--out", type=Path, help="Output .pto path (default: foo.pto or foo.<kernel>.pto)")
    ap.add_argument("--no-host-spec", dest="host_spec", action="store_false", default=True)
    args = ap.parse_args()

    if not args.py.exists():
        print(f"error: kernel file not found: {args.py}", file=sys.stderr)
        return 2

    try:
        out_path = binding.write_pto(args.py, kernel=args.kernel, out_path=args.out, universal=args.host_spec)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"OK: wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
