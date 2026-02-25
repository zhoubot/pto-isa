# PTO Bytecode (PTO-BC) — Decisions Log

This file records design decisions, one question at a time, to converge on a Sail-formal bytecode spec.

## Open questions

- QBC2: Do we require **lossless round-trip** back to textual `.pto`?
  - A) Yes: decoder must reproduce an equivalent `.pto` where re-parsing yields identical IR (allowing non-semantic formatting differences).
  - B) No: only require semantic equivalence for PTOAS consumption; debug names/pretty-print not required.

- QBC3: Opcode strategy for operations (`pto.*`, `arith.*`, `scf.*`):
  - A) Generic ops only (encode op name as string + operand/result counts/types)
  - B) Fixed opcode table for common ops + generic escape for unknown ops

(older questions)
- (QBC1 answered)

## Answered

- QBC1: **A** — canonical encoding target is PTODSL-emitted textual `.pto` (high-level PTO-IR).
- QBC2: **A** — require lossless (re-parse stable) round-trip back to textual `.pto`.
- QBC3: **B** — fixed opcode table for common ops + generic escape for unknown ops.
- QBC4: **C** — mixed: structured encodings for common PTO types/attrs + string fallback for unknown.
- QBC5: SSA naming/debug: follow industry bytecode practice — do **not** require storing `%0/%1` names; add an **optional DebugInfo section** to preserve names/locations when desired.
- QBC6: **C** — DebugInfo supports full fidelity: function/op/value names + source locations (line/col ranges) + optional raw text snippets.
- QBC7: **A** — bytecode supports structured control flow only (scf.for/scf.if + pto.section regions), matching PTODSL output.
- QBC8: **A** — integers/constants use (s/u)LEB128 varint encoding (industry standard).
- QBC9: **A** — no compression in v0 (keep container simple / mmap-friendly).
- QBC10: **B** — support import/export in v0 (forward-looking linking model).
- QBC11: **B** — symbol model includes functions + globals (const pool / global vars) for future-proofing.
- QBC12: **A** — v0 globals: const-pool only (no RO/RW global tensors yet).
- QBC13: **B** — allow const-pool entries for index vectors (shape/stride/offset/size lists) and reference by id.
- QBC14: **C** — fixed opcode coverage aims for full PTO-IR (pto + arith + scf), following the PTO IR docs as the source.
- QBC15: **A** — v0 semantics require only the PTO-IR-approved `arith/scf` subset, but the encoding reserves space to grow toward full-set coverage.
- QBC16: **A** — opcode space is segmented by dialect, with `pto.*` split into L1/L2 ranges:
  - 0x0000–0x0FFF: pto L1 ops
  - 0x1000–0x1FFF: pto L2 ops
  - 0x2000–0x3FFF: arith ops
  - 0x4000–0x5FFF: scf ops
  - 0xE000–0xEFFF: experimental
  - 0xF000–0xFFFE: vendor/private
  - 0xFFFF: GENERIC_OP escape
- QBC17: **B** — `index_width` is stored in the module header (32/64), default 64.
- QBC18: **A** — arithmetic uses two's-complement modular semantics at the declared bitwidth (i32/i64/index_width) unless an op explicitly specifies otherwise.
- QBC19: **A** — attributes must be preserved losslessly; unknown attrs are stored opaquely (verbatim text blob or canonical string) so decoding can reproduce `.pto` attrs.
- QBC20: **A** — fixed-width fields are little-endian.
- QBC21: **A** — use a mandatory string table; all strings are interned and referenced by `string_id`.
- QBC22: **A** — fixed section order, each section appears at most once.
- QBC23: **C** — section order is: Header → Tables (strings/types/attrs/const) → Module → DebugInfo → Extra/Unknown.
- QBC24: **C** — TypeTable uses structured encoding + MLIR-assembly string backup for lossless; over time, newly frozen types should migrate from string fallback to structured encoding.
- QBC25: **C** — AttrTable uses structured encoding + MLIR-assembly string backup for lossless; newly frozen attrs should migrate from string fallback to structured encoding.
- QBC26: **B** — ConstPool supports integer + floating constants; floating covers PTO-relevant low-precision formats (fp16/bf16/fp32 and fp8/fp4 families per ISA), with extension space.
- QBC27: **A** — floating constants are stored as dtype-tagged raw bitpatterns (bytes), to guarantee lossless fp8/fp4.
- QBC28: **C** — mixed enum encoding in structured types: high-frequency enums (loc/blayout/slayout/dtype) are numbered; less common fields may use string/opaque, with asm backup.
- QBC29: **C** — dtype id table is the union of ISA-doc + C++ types, with explicit profile/target support matrix (A2/A3/A5, CPU-sim).
- QBC30: **C** — carry a small `profile_id` in the header for fast validation, while preserving `pto.device-spec` and other target info losslessly in attrs.
- QBC31: **A** — profile_id initial enum: 0=unspecified, 1=cpu-sim, 2=a2a3, 3=a5 (extendable).
- QBC32: **A** — preserve `.pto` modeling: TileBuf type includes static/dynamic marker for v_row/v_col; dynamic valid is encoded on `alloc_tile` op fields.
- QBC33: **A** — `pto.section.*` is a structured control op in bytecode (dedicated opcode + region), with section_kind encoded as enum.
- QBC34: **A** — value IDs are function-global (monotonic) across nested regions.
- QBC35: **A** — blocks explicitly declare block-args with type_ids; block-args consume global value_ids.
- QBC36: **A** — module contains a FunctionTable (name_id + signature + import/export flags), followed by function bodies in table order.
- QBC37: **A** — TypeTable includes FuncType (arg type_ids + result type_ids); FunctionTable references func_type_id.
- QBC38: **A** — GENERIC_OP escape stores full schema: op-name + operand/result counts + type_ids + regions + attr_id.
- QBC39: **B** — known ops use compact encoding per opcode schema; unknown ops use GENERIC_OP full encoding.
- QBC40: **A** — every op record carries an `attr_id` (0 = none) to preserve attrs losslessly.
- QBC41: **C** — mixed result-type encoding: infer for most known ops; explicit result type_ids only when schema cannot infer.
- QBC42: **A** — `arith.constant` always references ConstPool (const_id), enabling dedup for frequent scalars.
- QBC43: **A** — ConstPool dedup key is exact: (type_id, raw_bits/bytes) equality.
- QBC44: **A** — `arith.index_cast` is a fixed opcode and carries an explicit result type_id.
- QBC45: **A** — `arith.cmpi` predicate is encoded as a compact enum field (not as a string/attr).
- QBC46: **A** — `scf.for` uses a dedicated compact encoding (lb/ub/step value_ids + region with iv block-arg).
- QBC47: **A** — `scf.if` uses dedicated encoding: cond value_id + then_region + else_region (else may be empty).
- QBC48: **A** — `pto.record_event`/`pto.wait_event` are fixed opcodes with compact enum fields (src_op, dst_op, event_id).
- QBC49: **A** — event_id encoded as u8 (0..255), reserving low ids for standard EVENT_IDx.
- QBC50: **C** — opcode schemas are specified in the standard, plus an optional in-file schema extension section for experimental/vendor opcodes.
- QBC51: **A** — module carries a module-level `attr_id` referencing AttrTable (lossless), with optional DebugInfo snippets.
- QBC52: **A** — FunctionTable entries carry `func_attr_id` (0 = none) for lossless function-level attrs.
- QBC53: **B** — PTO op variants are encoded as opcode families with a compact variant-enum field (not separate opcodes).
- QBC54: **A** — variant enums use u8; low range standardized, high range reserved/vendor.
- QBC55: **C** — view ops may encode shape/stride/offset/size lists either as value_id vectors (pto-like) or as const_vector_id (compact), but decoding back to `.pto` normalizes to value_id lists.
- QBC56: **A** — const-pool index vectors have element type `index` (width from header).
- QBC57: **A** — v0 `pto.ptr<dtype>` has no explicit address-space; reserve extension tags for future spaces.
- QBC58: **C** — tensor_view/partition_view support variable rank (varint length), while profiles may validate rank<=5.
- QBC59: **A** — static partition-view dimensions are encoded as u16.
- QBC60: **B** — Sail model covers format + validation (decode + invariants + type/schema checks), not full execution semantics in v0.
- QBC61: **A** — Sail parsing uses cursor + read_* functions.
- QBC62: **A** — strict header validation: magic/version/flags must be recognized; unknown flags rejected in v0.
- QBC63: **A** — unknown sections allowed only in the trailing Extra area; decoder skips but preserves bytes for round-trip.
- QBC64: **A** — DebugInfo uses sparse tables (ValueNames, Locations, Snippets) keyed by value_id/op_id.
- QBC65: **B** — include a FileTable (file_id -> path(string_id) + optional hash) for source locations.
- QBC66: **A** — define a per-function op_id as the preorder op sequence index (covers ops with/without results).
- QBC67: **A** — op_id order is preorder DFS: number an op, then recursively number ops in its regions/blocks in source order.
- QBC68: **A** — source locations use 1-based (line, col) coordinates.
- QBC69: **A** — source ranges are half-open: [start, end) (end position is exclusive).
- QBC70: **A** — specify a canonical `.pto` printer for bytecode→text round-trip (formatting + ordering rules fixed).
- QBC71: **A** — canonical SSA names are generated from value_id order: %0..%N.
- QBC72: **B** — canonical printer uses deterministic `%c...` aliases for constants (derived from the printed immediate, e.g. `%c32`, `%c0x3f800000`).
- QBC73: **B** — canonical printer prints `arith.constant` immediates in bitpattern form (lossless for fp8/fp4).
- QBC74: **A** — canonical printer sorts attributes by key lexicographically.
- QBC75: **A** — bitpattern printing uses `0x` hex, width matches dtype bits, bytes interpreted little-endian.
- QBC76: **A** — ConstPool floating constants carry `dtype_id` (from the dtype table), not full type_id.
- QBC77: **A** — OpcodeSchema extension section is minimal: enough to decode compact encodings (arity/flags/result-type-mode), not a full constraint language.
- QBC78: **B** — v0 assigns fixed opcodes for the full PTO-IR op set (per docs), not just a staged subset.
- QBC79: **B** — opcode table is generated from repo docs (inputs locked, script versioned) to guarantee stability.
- QBC80: **A** — opcode generation uses normalized family names + variant lists (e.g., `tmatmul` family with variant=bias/acc), consistent with QBC53.
- QBC81: **A** — Sail formalization includes an explicit opcode→schema table (generated Sail source included), for full decode+validation.
- QBC82: **A** — versioning is monotonic: assigned ids/opcodes/enums never change; new versions only append/extend.
- QBC83: **A** — PTO-BC decoder/validator should not depend on MLIR runtime (self-describing format, with Sail spec).
