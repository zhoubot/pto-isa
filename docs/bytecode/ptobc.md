# ptobc — encoding rules (tool behavior)

This document describes the **actual encoding/decoding behavior** of the `ptobc` tool in this repository.

- **Normative format**: see [`docs/bytecode/pto-bc.md`](pto-bc.md).
- This file is **tool-specific**: it explains how `ptobc` maps MLIR/PTO-IR to PTO-BC v0 fields, what is deterministic, what is optional, and what is currently unsupported.

> TL;DR: `ptobc encode` parses `.pto` with the PTO dialect (from PTOAS), then emits **compact known-op PTO-BC v0** by default (no GENERIC fallback unless explicitly enabled). `ptobc decode` materializes an MLIR module and prints a **canonical** `.pto`.

---

## 1. Tool entry points

`ptobc` currently supports:

```bash
ptobc encode input.pto -o out.ptobc
ptobc decode in.ptobc -o out.pto
```

### Environment switches

Encoding:

- `PTOBC_ALLOW_GENERIC=1`
  - Allow fallback to `opcode==0xFFFF (GENERIC_OP)` for ops that are not in the v0 opcode table.
  - Default is **off** (strict): if an op is unknown, encoding fails.

- `PTOBC_EMIT_DEBUGINFO=1`
  - Emit optional `DEBUGINFO` section (FileTable + ValueNames + OpLocations), best-effort.

Decoding / printing:

- `PTOBC_PRINT_LOC=1`
  - Print `loc(...)` in decoded textual `.pto` (parseable form).

- `PTOBC_PRINT_GENERIC=1`
  - Force MLIR **generic** printing (quoted op names). Mostly for debugging.

- `PTOBC_PRINT_PRETTY=1`
  - Keep MLIR’s default float printing (non-canonical); otherwise scalar floats are forced to hex bitpattern form.

---

## 2. Version + tables used

### 2.1 PTO-BC version

`ptobc` currently writes **PTO-BC v0** only:

- header `version = 0`
- header `flags = 0`
- fixed section order as in the spec

### 2.2 v0 opcode/schema table (generated)

Compact encoding is table-driven.

The generator [`docs/bytecode/tools/gen_v0_tables.py`](tools/gen_v0_tables.py) produces:

- `docs/bytecode/generated/opcodes_v0.md` (human-readable)
- `tools/tools/sail/generated/pto_bc_opcodes_v0.tools/sail` (Sail decoder schema)
- `tools/ptobc/generated/ptobc_opcodes_v0.h` (**C++ schema header used by ptobc**)

The header provides:

- op name → `(opcode, variant_u8)` (`lookupOpcodeAndVariantByFullName`)
- opcode → schema (`operand_mode`, `imm_kind`, …)
- by-variant operand counts for matmul/gemv families

---

## 3. ID assignment rules

These rules affect determinism of the binary and of DebugInfo.

### 3.1 Function order (`func_id`)

`func_id` is the **0-based index** of top-level `func.func` ops in the module, in source order:

```cpp
for (auto f : module.getOps<mlir::func::FuncOp>()) { ... }
```

### 3.2 Value numbering (`value_id`)

Value IDs are **function-global** and assigned monotonically across the entire function body, including nested regions:

1) Block arguments (in block order) consume the next IDs.
2) Each op consumes IDs for its results (in result order).
3) Nested regions continue the same global counter.

This matches the spec section “Value numbering (function-global)”.

### 3.3 Op numbering (`op_id`) for DebugInfo

When `PTOBC_EMIT_DEBUGINFO=1`, `ptobc` numbers ops by **preorder DFS** per function:

- number an op
- then recursively number ops in its regions/blocks in source order

Only `FileLineColLoc` is currently recorded into `OpLocations` (best-effort).

---

## 4. STRINGS / TYPES / ATTRS tables

### 4.1 `string_id`

Strings are interned on demand.

- `string_id` values are not guaranteed stable across encodes unless the IR traversal order is stable.
- This does **not** affect semantic correctness (IDs are internal), but it means two encodes of semantically identical IR might differ in raw bytes unless upstream printing/traversal is identical.

### 4.2 `type_id` / `attr_id`

`ptobc` currently stores types/attributes in the v0 tables primarily as **MLIR asm strings** (opaque tag with `has_mlir_asm=1`).

- `type_id = 0` is reserved for “none” (so real types start at 1)
- `attr_id = 0` means “no attrs” (empty dict)

---

## 5. CONSTPOOL (what ptobc emits)

`ptobc` implements ConstPool emission and uses it for `arith.constant` compact encoding.

Currently emitted ConstEntry tags:

- `0x01 ConstInt`:
  - payload: `type_id(uLEB128)` + `value(sLEB128)`
  - used for scalar integer / index constants

- `0x02 ConstFloatBits`:
  - payload: `dtype_id(uLEB128)` + `byte_len(uLEB128)` + `bytes[byte_len]` little-endian
  - used for scalar float constants

Deduplication:

- constants may be deduplicated by exact tag+payload byte equality.

---

## 6. Compact known-op encoding (current implementation)

### 6.1 Selection rule: known-op vs GENERIC_OP

When encoding an op:

1) If the op name is in the v0 table, emit **known-op** record.
2) Otherwise:
   - if `PTOBC_ALLOW_GENERIC=1`, emit `opcode=0xFFFF` GENERIC_OP.
   - else fail.

This matches the project requirement “v0 frozen table must be fully compact” (no silent generic fallback).

### 6.2 Variant families (`variant_u8`)

Family ops are encoded using `variant_u8` (u8) as defined by the generator.

Examples:

- `pto.section.vector` uses base opcode for `pto.section` + `variant_u8`.
- `pto.tmatmul.acc` similarly.

### 6.3 Operand modes

`ptobc` follows the v0 schema (`operand_mode`):

- `0x00 fixed`: emit exactly `num_operands` value_ids
- `0x01 by_variant`: emit operand list of length derived from `(opcode, variant)`
- `0x02 varcount`: emit `n(uLEB128)` then `n` value_ids
- `0x03 segmented`: **list_mode=0 only** implemented
  - immediates include `(list_mode, n1, n2)`
  - operands are encoded inline as `base + n1 + n2` value_ids
- `0x04 optmask2` (alloc_tile): encode a mask for `(valid_row?, valid_col?)` and then `row+col` operands

### 6.4 Immediate kinds (examples)

`ptobc` currently implements these `imm_kind` behaviors:

- `arith.cmpi` (`imm_kind=0x01`): predicate encoded as u8 for {eq, ne, slt, sle, sgt, sge}.
- `arith.constant` (`imm_kind=0x05`): emit `const_id(uLEB128)` pointing into CONSTPOOL.
  - the `value` attribute is removed from `attr_id` and reconstructed by decoder.
- `pto.record_event/pto.wait_event` (`imm_kind=0x02`): event3(u8,u8,u8)
  - taken from attrs: `src_op`, `dst_op`, `event_id`.
- `pto.make_tensor_view` / `pto.partition_view` (`imm_kind=0x06/0x07`): list_mode + segment lengths
  - `ptobc` uses `list_mode=0` (inline value_id lists) for now.
- `pto.alloc_tile` (`imm_kind=0x08`): optional-mask for `valid_row` / `valid_col`.

---

## 7. Decoding and canonical `.pto` printing

### 7.1 Decode materialization

`ptobc decode`:

- reads sections
- materializes an MLIR module
- reconstructs attributes that were moved into immediates (e.g. `arith.constant value`)
- optionally applies `DEBUGINFO` locations to ops

### 7.2 Canonical printer guarantees

Decoded `.pto` is printed via the canonical printer, with these stability rules:

- attribute dictionaries are sorted lexicographically by key
- SSA names are canonicalized:
  - non-constants: `%0..%N`
  - scalar constants: `%c...` (derived from printed immediate + type)
- scalar float constants are printed as hex bitpatterns: `0x... : f16/f32/f64`

These rules are meant to make the “decode → reparse” loop stable.

---

## 8. Stage9 end-to-end harness

A repository smoke test exists:

```bash
./docs/bytecode/tools/stage9_ptobc_e2e.sh
PTOBC_E2E_DEBUGINFO=1 ./docs/bytecode/tools/stage9_ptobc_e2e.sh
```

The harness runs:

- `.pto -> ptobc encode -> ptobc decode -> ptobc encode`

and reports PASS/FAIL.

---

## 9. Current limitations / TODO

- `list_mode=1` (const-vector-id mode) for segmented operands is not implemented.
- ConstPool tag `0x03 ConstIndexVec` is parsed in decoder but not emitted by encoder yet.
- Full coverage of all v0 opcodes is incremental; current compact encode/decode is sufficient for `docs/bytecode/samples/*.pto`.
