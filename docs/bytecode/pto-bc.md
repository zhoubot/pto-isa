# PTO-BC v0 — Bytecode format for PTO-IR (`.pto`)

This document is the **normative specification** for **PTO-BC v0**, a binary encoding of PTO-IR programs.

PTO-BC is designed to encode the **PTODSL-emitted textual `.pto`** (high-level PTO-IR), and be **independent of MLIR’s internal serialization**. A PTO-BC decoder/validator should **not** require linking the MLIR runtime.

Samples used to validate design assumptions live in: [`docs/bytecode/samples/`](samples/).

---

## 0. Goals and scope

### 0.1 Canonical encoded IR stage

- PTO-BC v0 encodes **PTODSL-emitted** `.pto` (high-level PTO-IR).
- PTO-BC v0 must support **lossless round-trip** back to textual `.pto` such that re-parsing yields an equivalent IR (formatting may differ).

### 0.2 Supported control flow

- Only **structured control flow** is required:
  - `scf.for`, `scf.if`
  - `pto.section.*` regions

### 0.3 Debug information

- Debug information is **optional**, but when present it supports full fidelity:
  - function/op/value names
  - file + 1-based (line, col) source ranges
  - optional raw text snippets

### 0.4 Compatibility and versioning

- Versioning is **monotonic**:
  - Assigned IDs (opcodes/enums/type tags) **never change**.
  - New versions **only append/extend**.

---

## 1. Primitive encodings

### 1.1 Endianness

- All fixed-width integers are **little-endian**.

### 1.2 LEB128 varints

Unless explicitly stated as fixed-width, integers are encoded as:

- `uLEB128`: unsigned LEB128
- `sLEB128`: signed LEB128

**Canonical LEB requirement:** encoders MUST use the shortest valid LEB128 encoding (no redundant continuation/sign-extension bytes). Decoders MAY reject non-canonical encodings.

### 1.3 Common ID types

All IDs are `uLEB128` unless stated:

- `string_id`
- `type_id`
- `attr_id`
- `const_id`
- `func_id`
- `value_id`
- `op_id`

### 1.4 Source locations

- Coordinates are **1-based** (line, col).
- Ranges are **half-open**: `[start, end)`.

---

## 2. Container format

### 2.1 File header

```
struct PTOBC_Header {
  u8  magic[6];     // ASCII: 'P' 'T' 'O' 'B' 'C' '\0'
  u16 version;      // little-endian; v0 = 0
  u16 flags;        // little-endian; v0 requires flags == 0
  u32 payload_len;  // little-endian; number of bytes following the header
}
```

Validation rules (v0 strict):

- `magic` MUST equal `"PTOBC\0"`.
- `version` MUST equal `0`.
- `flags` MUST equal `0`.
- File size MUST be `sizeof(PTOBC_Header) + payload_len`.

### 2.2 Section framing

The payload is a sequence of sections, each with a fixed-width header:

```
struct SectionHeader {
  u8  section_id;
  u32 section_len;   // little-endian
}
section_data = bytes[section_len]
```

Decoders MUST be able to skip sections using `section_len`.

### 2.3 Fixed section order (v0)

Sections appear **in fixed order**, and each section appears **at most once**.

Order:

1. `STRINGS`
2. `TYPES`
3. `ATTRS`
4. `CONSTPOOL`
5. `OPCODE_SCHEMA_EXT` (optional)
6. `MODULE`
7. `DEBUGINFO` (optional)
8. `EXTRA` (optional, may contain unknown sections)

Unknown sections are only permitted in the trailing **EXTRA** area and MUST be preserved for round-trip.

### 2.4 Section IDs

- `0x01` — `STRINGS`
- `0x02` — `TYPES`
- `0x03` — `ATTRS`
- `0x04` — `CONSTPOOL`
- `0x05` — `OPCODE_SCHEMA_EXT` (optional)
- `0x06` — `MODULE`
- `0x07` — `DEBUGINFO` (optional)
- `0x7F` — `EXTRA` (optional)

`0x80..0xFE` reserved for future standard sections.

---

## 3. String table (`STRINGS`)

### 3.1 Encoding

```
STRINGS {
  uLEB128 count;
  repeat count times:
    uLEB128 byte_len;
    u8 bytes[byte_len];   // UTF-8
}
```

- `string_id` is the 0-based index into this table.
- Encoders SHOULD intern/deduplicate strings.

Strings are used for:
- symbol names
- MLIR-assembly backups for types and attrs
- generic op names
- debug file paths and snippets

---

## 4. DType IDs (normative enum)

PTO-BC uses `dtype_id` (uLEB128) to tag element types in ConstPool and structured PTO types.

The `dtype_id` universe is the **union** of:
- ISA-documented element types, and
- types used in the C++ headers,

with an explicit profile/target support matrix (e.g., CPU-sim vs A2/A3 vs A5).

**Note:** The concrete v0 `dtype_id` list is generated from repo docs and is part of the v0 release artifacts. (See §10.)

---

## 5. Type table (`TYPES`)

### 5.1 Overview

PTO-BC v0 uses a mixed strategy:

- **Structured encoding** for common types.
- A mandatory **MLIR-assembly backup string** for lossless round-trip.
- Newly “frozen” types should migrate from string fallback to structured fields over time.

### 5.2 Type entry framing

```
TYPES {
  uLEB128 type_count;
  repeat type_count times:
    TypeEntry
}

TypeEntry {
  u8 tag;
  u8 flags;               // bit0: has_mlir_asm
  if flags.bit0:
    uLEB128 mlir_asm_string_id;
  // tag-specific payload follows
}
```

### 5.3 Structured type tags (v0)

#### 5.3.1 Opaque type

- `tag = 0x00`
- No structured payload is required.
- MUST include `mlir_asm_string_id`.

#### 5.3.2 Integer type

- `tag = 0x01`

Payload:

```
IntType {
  u8 bitwidth;   // e.g. 1, 8, 16, 32, 64
}
```

MLIR integers are treated as **signless**; signedness is determined by the operation.

#### 5.3.3 Index type

- `tag = 0x02`

No payload.

The bitwidth is given by `MODULE.index_width` (32 or 64).

#### 5.3.4 Float type

- `tag = 0x03`

Payload:

```
FloatType {
  uLEB128 dtype_id;  // must correspond to a floating format (f16/bf16/f32/.../fp8/fp4 families)
}
```

#### 5.3.5 PTO pointer type

- `tag = 0x10`

Payload:

```
PtoPtrType {
  uLEB128 elem_type_id;
}
```

v0 does not encode an address-space; future tags may extend this.

#### 5.3.6 PTO tensor view type

- `tag = 0x11`

Payload:

```
PtoTensorViewType {
  uLEB128 rank;
  repeat rank times:
    Dim
  uLEB128 elem_type_id;
}

Dim {
  u8 kind;       // 0 = DYN, 1 = STATIC
  if kind==1:
    uLEB128 value;
}
```

Profiles may validate `rank<=5`, but the encoding supports variable rank.

#### 5.3.7 PTO partition tensor view type

- `tag = 0x12`

Payload:

```
PtoPartitionTensorViewType {
  uLEB128 rank;
  repeat rank times:
    u16 dim;      // little-endian; static dims only
  uLEB128 elem_type_id;
}
```

#### 5.3.8 PTO tile buffer type

- `tag = 0x13`

Payload:

```
PtoTileBufType {
  u8  loc_id;
  uLEB128 elem_type_id;

  u16 rows;
  u16 cols;

  u8  vrow_mode;   // 0=static, 1=dynamic
  if vrow_mode==0: u16 vrow;

  u8  vcol_mode;   // 0=static, 1=dynamic
  if vcol_mode==0: u16 vcol;

  u8  blayout_id;
  u8  slayout_id;

  u16 fractal;
  u16 pad;
}
```

Notes:
- `loc_id/blayout_id/slayout_id` are numbered enums (high-frequency fields). Unknown values may be represented by `tag=0x00` + asm.
- Dynamic `valid_row/valid_col` values are carried by the `pto.alloc_tile` op fields (see §7).

#### 5.3.9 Function type

- `tag = 0x20`

Payload:

```
FuncType {
  uLEB128 num_args;
  repeat num_args times: uLEB128 arg_type_id;
  uLEB128 num_results;
  repeat num_results times: uLEB128 res_type_id;
}
```

---

## 6. Attribute table (`ATTRS`)

### 6.1 Overview

- Attributes MUST be preserved losslessly.
- Attr entries use structured encoding when available, with an MLIR-assembly backup string.

### 6.2 Encoding

```
ATTRS {
  uLEB128 attr_count;
  repeat attr_count times:
    AttrEntry
}

AttrEntry {
  u8 tag;
  u8 flags;                    // bit0: has_mlir_asm (MUST be 1 in v0)
  uLEB128 mlir_asm_string_id;  // required in v0
  // tag-specific structured payload (optional in v0)
}
```

`attr_id = 0` is reserved to mean **no attributes**. Therefore the first real attribute entry SHOULD start at index 1; encoders may include a dummy entry 0 or simply treat 0 as “none” without table storage.

---

## 7. Constant pool (`CONSTPOOL`)

### 7.1 Overview

ConstPool is used for:
- deduplicated scalar constants (including `arith.constant`)
- index vectors (shape/stride/offset/size lists)
- low-precision floating bitpatterns (fp8/fp4 families)

### 7.2 Encoding

```
CONSTPOOL {
  uLEB128 const_count;
  repeat const_count times:
    ConstEntry
}

ConstEntry {
  u8 tag;
  // tag-specific payload
}
```

Dedup rule (exact): the encoder MAY deduplicate constants; if it does, it MUST treat constants as equal only if all tag+payload bytes match exactly.

### 7.3 ConstEntry tags (v0)

#### 7.3.1 Integer-like scalar

- `tag = 0x01`

Payload:

```
ConstInt {
  uLEB128 type_id;  // IntType or IndexType
  sLEB128 value;    // canonical sLEB128
}
```

#### 7.3.2 Floating scalar (bitpattern)

- `tag = 0x02`

Payload:

```
ConstFloatBits {
  uLEB128 dtype_id;       // floating dtype
  uLEB128 byte_len;
  u8 bytes[byte_len];     // raw bitpattern, little-endian
}
```

This representation MUST be used for fp8/fp4 families to remain lossless.

#### 7.3.3 Index vector

- `tag = 0x03`

Payload:

```
ConstIndexVec {
  uLEB128 length;
  repeat length times:
    sLEB128 elem;   // element type is `index` (width from MODULE.index_width)
}
```

---

## 8. Opcode schema extension (`OPCODE_SCHEMA_EXT`, optional)

This optional section allows experimental/vendor opcodes to be decoded in compact form.

It is **minimal** (not a constraint language): it provides only the data needed to decode bytecode records.

```
OPCODE_SCHEMA_EXT {
  uLEB128 entry_count;
  repeat entry_count times:
    SchemaEntry
}

SchemaEntry {
  u16 opcode;                // little-endian
  uLEB128 op_name_string_id;  // for diagnostics / round-trip

  u8 flags;                  // bit0: has_variant_u8
  u8 result_type_mode;       // 0=infer, 1=explicit

  uLEB128 num_operands;      // fixed arity in v0 extension
  uLEB128 num_results;       // fixed arity
  uLEB128 num_regions;       // fixed arity
}
```

---

## 9. Module (`MODULE`)

### 9.1 Module header

```
MODULE {
  u8 profile_id;      // 0=unspecified, 1=cpu-sim, 2=a2a3, 3=a5
  u8 index_width;     // 32 or 64

  uLEB128 module_attr_id;   // 0 = none

  uLEB128 global_count;
  repeat global_count times:
    GlobalEntry

  uLEB128 func_count;
  repeat func_count times:
    FunctionDecl

  // Function bodies follow, in the same order as declarations,
  // for each non-import function.
  repeat func_count times:
    if FunctionDecl.flags.import==0:
      FunctionBody
}
```

### 9.2 Globals (v0: const-pool only)

```
GlobalEntry {
  uLEB128 name_string_id;
  u8 flags;              // bit0=import, bit1=export
  u8 kind;               // v0: 0x01 = constpool_ref
  uLEB128 payload;       // for kind=constpool_ref: const_id
}
```

### 9.3 Function declarations

```
FunctionDecl {
  uLEB128 name_string_id;
  uLEB128 func_type_id;   // TypeTable tag=0x20
  u8 flags;               // bit0=import, bit1=export
  uLEB128 func_attr_id;   // 0 = none
}
```

### 9.4 Function body

A function body is encoded as a single **top-level region**:

```
FunctionBody {
  Region
}
```

#### 9.4.1 Region encoding

```
Region {
  uLEB128 block_count;
  repeat block_count times:
    Block
}
```

#### 9.4.2 Block encoding

```
Block {
  uLEB128 num_block_args;
  repeat num_block_args times:
    uLEB128 arg_type_id;

  uLEB128 op_count;
  repeat op_count times:
    Op
}
```

Value numbering (function-global):

- Value IDs are assigned monotonically across the entire function, including nested regions.
- Block arguments consume the next `value_id` sequence in order.
- Each op consumes the next `value_id` sequence for its results, in order.

`op_id` numbering (for DebugInfo):

- `op_id` is assigned by **preorder DFS**: number an op, then recursively number ops in its regions/blocks in source order.

---

## 10. Operations (`Op`)

### 10.1 Opcode space

`opcode` is a `u16` (little-endian). Space is segmented:

- `0x0000–0x0FFF`: `pto` L1 ops
- `0x1000–0x1FFF`: `pto` L2 ops
- `0x2000–0x3FFF`: `arith` ops
- `0x4000–0x5FFF`: `scf` ops
- `0x6000–0xDFFF`: other dialects (e.g., minimal `func.return`), as frozen by the v0 generator
- `0xE000–0xEFFF`: experimental
- `0xF000–0xFFFE`: vendor/private
- `0xFFFF`: `GENERIC_OP` escape

The **v0 opcode table** is generated from repo docs (locked inputs + versioned script) and is a normative artifact.

### 10.2 Op record framing

All ops carry `attr_id` (0 = none).

#### 10.2.1 Known op (compact)

```
Op {
  u16 opcode;
  uLEB128 attr_id;

  // opcode-schema-defined immediate fields
  // opcode-schema-defined operands
  // optional explicit result type_ids (only when schema requires)
  // opcode-schema-defined regions
}
```

Notes:
- Result types are mostly inferred; certain ops require explicit result type IDs (e.g., `arith.index_cast`).
- PTO op variants are encoded as opcode families with a compact `variant_u8` field.
- Matmul/Gemv families (`pto.tmatmul*`, `pto.tgemv*`) use `operand_mode=by_variant`; operand counts for each variant are derived from PTOAS TableGen (`PTOOps.td`).
- `scf.for` is compact-encoded with fixed operands (lb, ub, step) and exactly 1 region.
- `scf.if` is compact-encoded with fixed operand (cond) and exactly 2 regions (then/else).
- Most PTO ISA ops (pto-l2) are compact-encoded as **fixed-arity** DPS ops:
  - operand count is `|ins| + |outs|` from the ISA docs (`pto.xxx ins(...) outs(...)`)
  - they have no regions and no extra immediates beyond `attr_id`.

#### 10.2.2 Generic op escape

If `opcode == 0xFFFF`:

```
GenericOp {
  u16 opcode = 0xFFFF;
  uLEB128 attr_id;

  uLEB128 op_name_string_id;

  uLEB128 num_results;
  repeat num_results times:
    uLEB128 result_type_id;

  uLEB128 num_operands;
  repeat num_operands times:
    uLEB128 operand_value_id;

  uLEB128 num_regions;
  repeat num_regions times:
    Region
}
```

This carries the full schema to enable lossless round-trip.

### 10.3 Selected immediate encodings (normative)

These are encoded as compact fields in known-op schemas:

- `arith.cmpi`: predicate is an enum `u8` (eq/ne/lt/le/gt/ge)
- `arith.constant`: `const_id(uLEB128)` (ConstPool index)
- `arith.select`: no immediate; fixed operands = (cond, true_value, false_value)
- `pto.record_event` / `pto.wait_event`: `src_op(u8)`, `dst_op(u8)`, `event_id(u8)`
- `pto.section` family: `variant_u8` encodes section kind (`cube`/`vector`) and the op carries one region.

View ops carry **segment metadata** as immediates:

- `pto.make_tensor_view`: `list_mode(u8)` + `nshape(uLEB128)` + `nstrides(uLEB128)`
- `pto.partition_view`: `list_mode(u8)` + `noffsets(uLEB128)` + `nsizes(uLEB128)`

`list_mode`:
- 0 = encode lists as value_id vectors (lengths given by the uLEB128 fields)
- 1 = encode each list as a single `const_vector_id` operand (2 operands total for the two lists)

Tile allocation carries an optional-operand mask:

- `pto.alloc_tile`: `opt_mask(u8)` where bit0=has_valid_row, bit1=has_valid_col

Synchronization:

- `pto.barrier`: fixed encoding, 0 operands
- `pto.record_event`/`pto.wait_event`: event triple immediates (src_op,dst_op,event_id)
- `pto.tsync`: operand_mode=varcount, encodes `n(uLEB128)` followed by `n` operands

---

## 11. DebugInfo (`DEBUGINFO`, optional)

DebugInfo is encoded as sparse tables.

```
DEBUGINFO {
  FileTable
  ValueNames
  OpLocations
  OpSnippets
}
```

### 11.1 FileTable

```
FileTable {
  uLEB128 file_count;
  repeat file_count times:
    uLEB128 path_string_id;
    u8 hash_kind;           // 0=none, 1=sha256
    if hash_kind!=0:
      uLEB128 hash_len;
      u8 hash_bytes[hash_len];
}
```

### 11.2 ValueNames

```
ValueNames {
  uLEB128 entry_count;
  repeat entry_count times:
    uLEB128 value_id;
    uLEB128 name_string_id;
}
```

### 11.3 OpLocations

```
OpLocations {
  uLEB128 entry_count;
  repeat entry_count times:
    uLEB128 op_id;
    uLEB128 file_id;

    uLEB128 start_line;
    uLEB128 start_col;
    uLEB128 end_line;
    uLEB128 end_col;
}
```

### 11.4 OpSnippets

```
OpSnippets {
  uLEB128 entry_count;
  repeat entry_count times:
    uLEB128 op_id;
    uLEB128 snippet_string_id;
}
```

---

## 12. Canonical `.pto` printer (bytecode → text)

A canonical printer is required for stable round-trip diffs.

Rules (v0):

1) **Attribute ordering**: print attrs sorted by key lexicographically.

2) **SSA naming**:
   - Non-constant SSA values are named deterministically by `value_id` order: `%0..%N`.
   - Constants use deterministic `%c...` aliases derived from the printed immediate.

3) **Constant printing**:
   - Integer-like constants (`i*`, `index`) MAY be printed in decimal.
   - Floating constants MUST be printed in **hex bitpattern** form `0x...` with width matching dtype bits, interpreting stored bytes as little-endian.

4) **Sections/regions**:
   - `pto.section.*` and `scf.*` regions are printed with fixed indentation and brace placement.

---

## 13. Release artifacts and generators

Because v0 freezes full opcodes and dtype universes, some spec components are **generated** from repo docs:

- v0 opcode table (including family/variant normalization)
- v0 dtype_id list + profile support matrix
- Sail include file for opcode→schema table

These artifacts MUST be versioned alongside the spec and used by both:

- a reference encoder/decoder
- the Sail formal model (decode + validation)
