# PTO ISA — Sail Formal Spec & Bytecode Design (Working Notes)

> Branch: `bytecode`
>
> Goal: write a Sail formal definition for PTO ISA and design a PTO ISA bytecode encoding.
>
> Repo anchors:
> - ISA index: `docs/isa/manifest.yaml`
> - C++ intrinsics (source of truth): `include/pto/common/pto_instr.hpp`
> - Tile / layouts: `include/pto/common/pto_tile.hpp`
> - PTO-AS (textual asm): `docs/grammar/PTO-AS.md`

## Q&A Decisions

- Q1 (scope): **B** — cover the full instruction set (≈116) while allowing abstract/"uninterpreted" placeholders for complex ops initially; **must keep extension space**.

(TODO) Q2 (truth source):
(TODO) Q3 (tile representation):
(TODO) Q4 (fp/math functions):
(TODO) Q5 (events/TSYNC modeling):
(TODO) Q6 (bytecode route):

## Plan (high level)

1. Define architectural state model (tiles, GM/memrefs, events/pipelines).
2. Define type system and value domains (dtypes, rounding/sat/cmp modes, valid region, undefined behavior policy).
3. Define instruction semantics by groups; allow "abstract" semantics where needed, but ensure:
   - decoding/typing is precise
   - operand constraints are checked
   - extension/versioning hooks exist
4. Define bytecode format:
   - stable opcode space
   - type table + attribute encoding
   - constant pool
   - SSA (or reg) value mapping
   - forward-compat extension sections
5. Build a small interpreter harness (optional) and conformance tests.
