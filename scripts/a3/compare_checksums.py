#!/usr/bin/env python3
import re
import sys


RUN_RE = re.compile(r"^\[run\]\s+.*\s+\(([^)]+)\)\s+memrefs=\d+")
OUT_RE = re.compile(r"^\s*\[out\]\s+(\S+)\s+bytes=\d+\s+checksum=0x([0-9a-fA-F]+)")


def parse_log(path):
    cur_prog = None
    out = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = RUN_RE.match(line.strip())
            if m:
                cur_prog = m.group(1)
                continue
            m = OUT_RE.match(line.strip())
            if m and cur_prog:
                key = (cur_prog, m.group(1))
                out[key] = m.group(2).lower()
    return out


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <cpu_log> <npu_log>")
        return 2

    cpu = parse_log(sys.argv[1])
    npu = parse_log(sys.argv[2])

    missing = []
    mismatched = []
    for key, cpu_sum in cpu.items():
        npu_sum = npu.get(key)
        if npu_sum is None:
            missing.append(key)
            continue
        if npu_sum != cpu_sum:
            mismatched.append((key, cpu_sum, npu_sum))

    extra = [k for k in npu.keys() if k not in cpu]

    if missing:
        print("Missing in NPU log:")
        for prog, mem in missing:
            print(f"  - {prog}:{mem}")
    if extra:
        print("Extra in NPU log:")
        for prog, mem in extra:
            print(f"  - {prog}:{mem}")
    if mismatched:
        print("Checksum mismatches:")
        for (prog, mem), cpu_sum, npu_sum in mismatched:
            print(f"  - {prog}:{mem} cpu=0x{cpu_sum} npu=0x{npu_sum}")

    if missing or extra or mismatched:
        return 1
    print("Checksums match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
