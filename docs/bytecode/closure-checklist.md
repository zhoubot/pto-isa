# PTO-BC v0 — Closure Checklists (Compact Encoding)

This document tracks *separate* closure milestones to make sure **every op** is encoded/decoded/validated in **compact form** (no GENERIC_OP fallback), and the spec + generator + Sail stay consistent.

Status legend:
- [ ] TODO
- [~] DOING
- [x] DONE

## Stage 0 — “All opcodes have a compact schema entry” (baseline)

- [x] `docs/bytecode/generated/opcodes_v0.md` generated without false-positive ops (no `pto.device`, no `pto.op`).
- [x] `tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail` contains **one schema entry per frozen opcode**.
- [x] Count check: `schema_entries == opcode_count`.

Acceptance:
- A script check passes:
  - `opcode_count == schema_entries`.

---

## Stage 1 — Freeze the *authoritative* opcode/schema source

Goal: opcode/schema generation is *stable and reproducible*.

- [ ] Decide authoritative sources for arity/segments:
  - [ ] PTOAS TableGen: `PTOAS/include/PTO/IR/PTOOps.td` (operand segments / optional / variadic)
  - [ ] ISA doc: `docs/isa/*.md` DPS lines for pto.* instruction ops (ins/outs)
  - [ ] PTO IR doc: `docs/ir/PTO-IR-ops.md` for non-ISA ops
- [ ] Generator only uses these inputs (no ad-hoc regex over random docs).
- [ ] Document the generator contract in `docs/bytecode/pto-bc.md`.

Acceptance:
- Running generator twice produces identical `opcodes_v0.md` and identical Sail tables.

---

## Stage 2 — Compact schema closure for PTO-IR structural ops (pto-l1)

### 2.1 `pto.make_tensor_view`
- [x] Encode **operand segments** exactly:
  - ptr (1)
  - shape (variadic)
  - strides (variadic)
  - layout attr (attr only; not operand)
- [x] Segment-length encoding: `list_mode(u8) + nshape(uLEB128) + nstrides(uLEB128)` (compact immediate).
- [x] Sail decode validates segment structure (via operand_mode=segmented + imm_kind=6).

Acceptance:
- PTODSL sample `.pto` for add/relu/matmul decodes in compact mode.

### 2.2 `pto.partition_view`
- [x] Encode segments: source (1), offsets (variadic), sizes (variadic).
- [x] Segment-length encoding: `list_mode(u8) + noffsets(uLEB128) + nsizes(uLEB128)` (compact immediate).
- [ ] Sail validates offsets/sizes length match rank constraints (best-effort; not required for v0 decoding).

### 2.3 `pto.alloc_tile`
- [x] Encode optional operands: valid_row?, valid_col? using `opt_mask(u8)` immediate (bit0=row, bit1=col).

### 2.4 `pto.get_block_* / get_subblock_*`
- [x] Result type is losslessly carried via explicit `result_type_id` (result_type_mode=explicit).

---

## Stage 3 — Compact schema closure for `arith.*`

- [x] `arith.constant`: immediate uses const_id (uLEB128) + explicit result type_id.
- [x] `arith.index_cast`: explicit result type_id.
- [x] `arith.cmpi`: predicate u8 immediate.
- [x] Remaining arith ops used by PTODSL samples:
  - addi/subi/muli/ceildivsi/minui/select
- [x] Static schema check script passes: `docs/bytecode/tools/check_stage3_arith.py`.

Acceptance:
- All `docs/bytecode/samples/*.pto` use only arith ops with compact schemas (no GENERIC_OP fallback).

---

## Stage 4 — Compact schema closure for `scf.*`

- [x] `scf.for`: fixed fields (lb/ub/step) + region with iv block-arg.
- [x] `scf.if`: cond + then_region + else_region.
- [x] `scf.yield`: variadic terminator encoded via operand_mode=varcount.
- [x] Static schema check script passes: `docs/bytecode/tools/check_stage4_scf.py`.

Acceptance:
- PTODSL samples containing loops/ifs use only `scf.for`/`scf.if` with compact schemas.

---

## Stage 5 — Compact schema closure for tile/memory ops (pto-l2 ISA)

Goal: all ISA ops are compactly encodable using **DPS arity** from docs.

- [x] Build a reliable DPS arity table: parse `docs/isa/*.md` lines `pto.xxx ins(...) outs(...)`.
- [x] For each fixed-arity ISA op in opcode table:
  - [x] Determine fixed operand count = ins+outs (DPS)
- [~] Ops with optional/variadic operands are handled in dedicated stages:
  - [~] `pto.tsync` (Stage 7)
  - [~] matmul/gemv families (Stage 6)
- [x] Static schema check script passes: `docs/bytecode/tools/check_stage5_isa.py`.

Acceptance:
- All DPS ISA ops (excluding Stage6/Stage7 families) have operand_mode=fixed and num_operands matching DPS.
- Generated report: `docs/bytecode/generated/isa_dps_arity_v0.md`.

---

## Stage 6 — Compact schema closure for matmul/gemv families

Goal: reflect *actual PTOAS op spellings* and operand segments.

- [x] Align family design with PTOAS ops (discovered from `PTOAS/include/PTO/IR/PTOOps.td`):
  - `pto.tmatmul` family (base + discovered suffix variants)
  - `pto.tmatmul.mx` family if present (base + discovered suffix variants)
  - `pto.tgemv` family (base + discovered suffix variants)
  - `pto.tgemv.mx` family if present (base + discovered suffix variants)
- [x] `.mx` treated as a **separate family base** when present.
- [x] by-variant operand counts match PTOAS TableGen op arguments.
- [x] Static check script passes: `docs/bytecode/tools/check_stage6_matmul_gemv.py`.

Acceptance:
- All discovered matmul/gemv family variants have compact by-variant operand counts consistent with TableGen.

---

## Stage 7 — Events / barriers / tsync

- [x] `pto.record_event`/`pto.wait_event`: encode 3 u8 immediates (src_op,dst_op,event_id).
- [x] `pto.barrier`: compact encoding with 0 operands.
- [x] `pto.tsync`: compact encoding with operand_mode=varcount (uLEB128 count + that many operands).
- [x] Stage7 sample: `docs/bytecode/samples/sync_stage7.pto`.
- [x] Static schema check script passes: `docs/bytecode/tools/check_stage7_sync.py`.

Acceptance:
- Stage7 ops have compact schemas and are covered by `sync_stage7.pto`.

---

## Stage 8 — DebugInfo + canonical `.pto` printer

- [x] DebugInfo tables: ValueNames / Locations / Snippets + FileTable.
- [x] Location conventions: 1-based, half-open.
- [x] Canonical printer:
  - [x] `%c...` deterministic constant aliases
  - [x] bitpattern printing `0x...`
  - [x] attrs sorted by key

Acceptance:
- bytecode → `.pto` print → parse produces stable IR (re-parse stable).

---

## Stage 9 — Final integration tests

- [~] Add a test harness (Python or C++) that:
  - [~] builds `.pto` samples via PTODSL
  - [~] encodes to PTO-BC
  - [ ] decodes with Sail model (or reference decoder)
  - [ ] prints back to `.pto`
  - [ ] re-parses via PTOAS/MLIR (optional) and checks structural equality

- [x] Minimal container harness exists (reference-level):
  - `docs/bytecode/tools/encode_ptobc_ref.py`
  - `docs/bytecode/tools/stage9_harness.py`

Acceptance:
- CI-like script returns 0 with logs.
