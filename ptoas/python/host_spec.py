from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal


TensorRole = Literal["in", "out", "inout"]


@dataclass(frozen=True)
class HostTensorArg:
    # Argument order is implied by list position (matches %argN).
    dtype: str
    shape: tuple[int, int]
    role: TensorRole = "in"
    # Optional view metadata; used by host array generator for non-ND tensors (e.g. DN).
    # When omitted, callers should assume a contiguous ND buffer with default row-major strides.
    layout: str = "ND"
    stride: tuple[int, int] | None = None


@dataclass(frozen=True)
class HostSpec:
    """
    Host-side metadata embedded in a `.pto` file for cross-platform testing/running.

    This is intentionally small and JSON-serializable so the same `.pto` can drive:
      - CPU compilation + run (reference)
      - NPU compilation + run, compared against CPU
    """

    args: tuple[HostTensorArg, ...]
    seed: int = 0
    block_dim: int = 1
    kernel_name: str = "pto_kernel"

    def to_dict(self) -> dict:
        return {
            "kernel_name": str(self.kernel_name),
            "seed": int(self.seed),
            "block_dim": int(self.block_dim),
            "args": [
                {
                    "dtype": str(a.dtype),
                    "shape": [int(a.shape[0]), int(a.shape[1])],
                    "role": a.role,
                    "layout": str(a.layout),
                    "stride": ([int(a.stride[0]), int(a.stride[1])] if a.stride is not None else None),
                }
                for a in self.args
            ],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    def output_indices(self) -> list[int]:
        return [i for i, a in enumerate(self.args) if a.role in ("out", "inout")]


_BEGIN = "; PTO_HOST_SPEC_BEGIN v1"
_END = "; PTO_HOST_SPEC_END"


def encode_host_spec(spec: HostSpec) -> str:
    payload = {
        "kernel_name": spec.kernel_name,
        "seed": int(spec.seed),
        "block_dim": int(spec.block_dim),
        "args": [
            {
                "dtype": a.dtype,
                "shape": [int(a.shape[0]), int(a.shape[1])],
                "role": a.role,
                "layout": str(a.layout),
                "stride": ([int(a.stride[0]), int(a.stride[1])] if a.stride is not None else None),
            }
            for a in spec.args
        ],
    }
    # Keep this readable in diffs by formatting with 2-space indents.
    body = json.dumps(payload, sort_keys=True, indent=2)
    lines = [_BEGIN] + [f"; {ln}" for ln in body.splitlines()] + [_END, ""]
    return "\n".join(lines)


def prepend_host_spec_to_pto(*, pto: str, spec: HostSpec) -> str:
    return encode_host_spec(spec) + pto.lstrip()


def parse_host_spec_from_pto(pto: str) -> HostSpec | None:
    lines = pto.splitlines()
    try:
        i0 = next(i for i, ln in enumerate(lines) if ln.strip() == _BEGIN)
        i1 = next(i for i, ln in enumerate(lines) if i > i0 and ln.strip() == _END)
    except StopIteration:
        return None

    json_lines: list[str] = []
    for ln in lines[i0 + 1 : i1]:
        s = ln.strip()
        if not s.startswith(";"):
            continue
        s = s[1:].lstrip()
        json_lines.append(s)
    payload = json.loads("\n".join(json_lines))

    args: list[HostTensorArg] = []
    for a in payload.get("args", []):
        shape = a["shape"]
        stride = a.get("stride", None)
        stride2 = None
        if stride is not None:
            stride2 = (int(stride[0]), int(stride[1]))
        args.append(
            HostTensorArg(
                dtype=a["dtype"],
                shape=(int(shape[0]), int(shape[1])),
                role=a.get("role", "in"),
                layout=str(a.get("layout", "ND")),
                stride=stride2,
            )
        )

    return HostSpec(
        args=tuple(args),
        seed=int(payload.get("seed", 0)),
        block_dim=int(payload.get("block_dim", 1)),
        kernel_name=str(payload.get("kernel_name", "pto_kernel")),
    )


def infer_host_spec_from_pto(*, pto: str) -> HostSpec:
    """
    Best-effort inference for older `.pto` files that don't embed a host spec.

    This scans `pto.make_tensor_view %argN, dtype=..., shape=[H,W] ...` and:
      - orders args by N
      - marks the last arg as output
    """
    pat = re.compile(
        r"%[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*pto\.make_tensor_view\s+%arg(\d+)\s*,\s*dtype=([a-z0-9]+)\s*,\s*"
        r"shape=\[(\d+),(\d+)\]\s+strides=\[(\d+),(\d+)\]\s*,\s*layout=([A-Z0-9_]+)"
    )
    found: dict[int, HostTensorArg] = {}
    for m in pat.finditer(pto):
        idx = int(m.group(1))
        dt = m.group(2)
        h = int(m.group(3))
        w = int(m.group(4))
        s0 = int(m.group(5))
        s1 = int(m.group(6))
        layout = str(m.group(7))
        stride = None if (layout == "ND" and s0 == w and s1 == 1) else (s0, s1)
        found[idx] = HostTensorArg(dtype=dt, shape=(h, w), role="in", layout=layout, stride=stride)
    if not found:
        raise ValueError("failed to infer host args from .pto (no pto.make_tensor_view %argN found)")

    args: list[HostTensorArg] = []
    for i in sorted(found):
        args.append(found[i])
    if args:
        args[-1] = HostTensorArg(dtype=args[-1].dtype, shape=args[-1].shape, role="out")
    return HostSpec(args=tuple(args), seed=0, block_dim=1, kernel_name="pto_kernel")
