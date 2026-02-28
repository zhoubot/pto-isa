# 7. Instruction Families and Contracts

This chapter defines family-level normative contracts for PTO instructions. Per-op normative details remain in `docs/isa/*.md`.

## 7.1 Scope

This chapter covers:

- **Family taxonomy**: Instruction groupings by functionality
- **Family contracts**: Common requirements for all instructions
- **Valid-region rule**: How tile operations handle partial tiles
- **Per-op documentation**: Template for individual instruction pages
- **Synchronization policy**: Keeping documentation in sync with implementation

## 7.2 Family Taxonomy

PTO instruction families are organized by functionality:

### 7.2.1 Family Categories

| Category | Description | Examples |
|----------|-------------|----------|
| Synchronization | Resource binding and ordering | TSYNC, TASSIGN |
| Elementwise | Tile-tile arithmetic | TADD, TMUL, TCMP |
| Tile-Scalar | Tile-immediate operations | TADDS, TMULS |
| Reduction | Axis-based reductions | TROWSUM, TCOLMAX |
| Memory | GM-Tile movement | TLOAD, TSTORE |
| Matrix | Matrix operations | TMATMUL, TGEMV |
| Layout | Data transformation | TEXTRACT, TTRANS |
| Complex | Specialized operations | TQUANT, TGATHER |

### 7.2.2 Detailed Taxonomy

1. **Synchronization and Resource Binding**
   - Event-based synchronization: TSYNC
   - Tile addressing: TASSIGN
   - Mode configuration: TSETHF32MODE, TSETTF32MODE

2. **Elementwise Tile-Tile Operations**
   - Arithmetic: TADD, TSUB, TMUL, TDIV, TMOD
   - Bitwise: TAND, TOR, TXOR, TSHL, TSHR
   - Comparison: TCMP, TMIN, TMAX
   - Mathematical: TLOG, TEXP, TSQRT, TRSQRT, TRECIP
   - Activation: TRELU, TPRELU, TLRELU, TSIGMOID, TTANH

3. **Tile-Scalar and Tile-Immediate Operations**
   - Arithmetic: TADDS, TSUBS, TMULS, TDIVS
   - Comparison: TCMPS
   - Selection: TSELS
   - Broadcast: TEXPANDS

4. **Axis Reduce and Expand Operations**
   - Row reduction: TROWSUM, TROWPROD, TROWMAX, TROWMIN
   - Column reduction: TCOLSUM, TCOLPROD, TCOLMAX, TCOLMIN

5. **Memory Operations**
   - Basic: TLOAD, TSTORE
   - Indexed: MGATHER, MSCATTER
   - Prefetch: TPREFETCH

6. **Matrix Multiply and GEMV Operations**
   - Basic: TMATMUL
   - Fused: TMATMUL_ACC, TMATMUL_BIAS, TMATMUL_MX
   - Vector: TGEMV, TGEMV_BIAS

7. **Data Movement and Layout Transforms**
   - Extract/Insert: TEXTRACT, TINSERT
   - Copy: TMOV
   - Transform: TTRANS, TRESHAPE
   - Fill: TFILL, TFILLPAD

8. **Irregular and Complex Operations**
   - Quantization: TQUANT, TDEQUANT
   - Sort: TSORT
   - Padding: TPAD
   - Debug: TPRINT

The source-synchronized inventory is maintained by `docs/isa/manifest.yaml`.

## 7.3 Common Family Contract

Every instruction family MUST define:

### 7.3.1 Required Contract Elements

| Element | Description |
|---------|-------------|
| Operand classes | Input types (tile, memory, scalar, event) |
| Result classes | Output types |
| Position rules | Which operands go where |
| Semantic domain | Valid-region handling |
| Constraints | dtype/layout/location/shape requirements |
| Ordering | Synchronization implications |
| Diagnostics | Error behavior for illegal use |
| Implementation-defined | Backend-specific boundaries |

### 7.3.2 Contract Example

For elementwise operations (TADD family):

```text
Family: Elementwise Arithmetic
Operands: (tile, tile) -> tile
Semantics: dst[r,c] = src0[r,c] + src1[r,c] for all r<Rv, c<Cv
Constraints:
  - All tiles must have same dtype
  - All tiles must have same Rv, Cv
  - Tile classes: Vec, Mat, Left, Right, Acc
Synchronization: Producer/consumer ordering via events
Diagnostics: Type mismatch, shape mismatch
```

## 7.4 Valid-Region-First Rule

### 7.4.1 Definition

Unless a specific instruction states otherwise:

- **Semantics are defined only on the operation's valid domain** (Rv x Cv)
- **Out-of-domain results are unspecified**
- Family contracts MUST state domain-compatibility rules for multi-input operations

### 7.4.2 Valid Region Behavior

```
Tile with Rv=8, Cv=16 (physical 16x16):
+---------------------------+
| Computed region (8x16)    |  <- Defined semantics
+---------------------------+
| Unspecified region (8x16) |  <- Undefined behavior
+---------------------------+
```

### 7.4.3 Multi-Input Domain Rules

When multiple tiles are operands:

| Operation Type | Domain Rule |
|---------------|--------------|
| Elementwise | All operands must have identical Rv, Cv |
| Reduction | Output Rv or Cv changes based on axis |
| Matrix Multiply | LHS Rv x RHS Cv -> Output Rv x Cv |

## 7.5 Family-Level Summaries

### 7.5.1 Synchronization and Resource Binding

**Family**: `TSYNC`, `TASSIGN`, `TSETHF32MODE`, `TSETTF32MODE`

**Characteristics**:
- Define ordering or state-configuration effects
- Must preserve architecture ordering semantics
- No data results (may produce events)

**Contract Requirements**:
- TSYNC: Must establish happens-before ordering
- TASSIGN: Must define tile-to-address mapping
- Mode instructions: Must configure precision/behavior

### 7.5.2 Elementwise and Scalar Variants

**Family**: Arithmetic, bitwise, compare, select, unary math

**Characteristics**:
- Per-element operations
- Tile-tile and tile-scalar variants
- Mode-specific constraints (rounding, saturation)

**Contract Requirements**:
- Must define per-element behavior
- Must specify mode attribute effects
- Must document dtype constraints

### 7.5.3 Reduce/Expand Families

**Family**: Row/column reductions, broadcast expansions

**Characteristics**:
- Axis-based operations
- Domain transformation (Rv x Cv -> 1 x Cv or Rv x 1)
- Mask patterns for selective computation

**Contract Requirements**:
- Must define axis semantics
- Must specify domain compatibility
- Must document output domain shape

### 7.5.4 Memory Families

**Family**: TLOAD, TSTORE, MGATHER, MSCATTER, TPREFETCH

**Characteristics**:
- Global memory ↔ Tile data movement
- Indexed and strided access variants
- Atomic operations for accumulation

**Contract Requirements**:
- Must define tile ↔ memory mapping
- Must specify indexing semantics
- Must document alignment requirements

### 7.5.5 Matrix Families

**Family**: TMATMUL, TMATMUL_*, TGEMV, TGEMV_*

**Characteristics**:
- Fused multiply-accumulate
- Multiple precision modes (FP32, FP16, TF32, INT8)
- Accumulator tile handling

**Contract Requirements**:
- Must define accumulation domain
- Must specify operand-role legality (Left/Right/Acc)
- Must document precision-mode interactions

### 7.5.6 Movement/Layout Families

**Family**: TEXTRACT, TINSERT, TTRANS, TRESHAPE, TMOV

**Characteristics**:
- Tile domain transformation
- Index remapping
- Shape reinterpretation

**Contract Requirements**:
- Must define index mapping
- Must preserve valid-domain semantics
- Must document layout constraints

### 7.5.7 Complex/Irregular Families

**Family**: TQUANT, TSORT, TPAD, TGATHER, TSCATTER

**Characteristics**:
- Specialized operations
- May have implementation-defined portions
- Complex semantics

**Contract Requirements**:
- Must explicitly identify implementation-defined portions
- Must document edge cases
- Must specify algorithm behavior

## 7.6 Documentation Contract for Per-Op Pages

Each per-instruction page SHOULD follow the template in Appendix B:

### 7.6.1 Required Sections

| Section | Description |
|---------|-------------|
| Syntax | Assembly and intrinsic syntax |
| Operands | Input and output operands with types |
| Semantics | Mathematical/algorithmic interpretation |
| Constraints | dtype/layout/location/shape rules |
| Diagnostics | Error conditions and messages |
| Implementation-defined | Backend-specific behavior |
| Compatibility | Version and profile notes |

### 7.6.2 Example Template

```markdown
# TADD - Tile Addition

## Syntax
PTO-AS: `%dst = tadd %src0, %src1 : ...`
Intrinsic: `void TADD(TileDST, TileSRC0, TileSRC1)`

## Operands
| Operand | Type | Description |
|---------|------|-------------|
| dst | Tile | Destination tile |
| src0 | Tile | First source tile |
| src1 | Tile | Second source tile |

## Semantics
For all r in [0, Rv), c in [0, Cv):
  dst[r,c] = src0[r,c] + src1[r,c]

## Constraints
- dtype: f32, f16, i32, i16
- Rv, Cv: All operands must match
- Tile class: Vec, Mat, Left, Right, Acc

## Diagnostics
- PTO-VAL-001: dtype mismatch
- PTO-VAL-002: shape mismatch

## Implementation-defined
- None

## Compatibility
- PTO-AS v1.0+
- All backend profiles
```

## 7.7 Coverage and Synchronization Policy

### 7.7.1 Synchronization Requirements

Family and instruction indexes MUST stay synchronized with:

| Source | Description |
|--------|-------------|
| `docs/isa/manifest.yaml` | Master instruction inventory |
| `include/pto/common/pto_instr.hpp` | C++ intrinsic declarations |
| `docs/tools/` | Generated index/matrix tooling |

### 7.7.2 Verification

Each release SHOULD validate:

- All instructions in manifest have corresponding documentation
- All intrinsic declarations match assembly syntax
- Instruction matrix is complete and accurate

### 7.7.3 Manual Updates

When adding new instructions:

1. Update `docs/isa/manifest.yaml`
2. Add documentation in `docs/isa/<instruction>.md`
3. Regenerate indexes and matrices
4. Verify all cross-references

## 7.8 Related Documentation

| Topic | Reference |
|-------|-----------|
| Instruction index | docs/isa/README.md |
| Instruction manifest | docs/isa/manifest.yaml |
| Template | Appendix B |
| ISA table | docs/PTOISA.md |
