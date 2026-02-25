# PTO Virtual Instruction Set Architecture Manual

## 0.1 Scope

This manual defines the architecture-level contract of the PTO Virtual Instruction Set Architecture (VISA).
It specifies what a conforming frontend, IR pipeline, backend, and runtime MUST preserve when executing PTO programs.

Per-instruction pages in `docs/isa/*.md` remain the canonical source for opcode-specific semantics.
This manual defines the system-level contract around those semantics.

## 0.2 Audience

This manual is intended for:

- compiler and IR engineers implementing PTO lowering pipelines
- backend engineers implementing target legalization and code generation
- kernel authors validating architecture-visible behavior
- simulator and conformance-test developers

## 0.3 Document conventions

This manual uses a PTX/Tile-IR-inspired structure while preserving PTO-specific architecture design.
Each chapter follows a normative pattern where applicable:

- scope
- syntax/form
- semantics
- constraints
- diagnostics
- compatibility

## 0.4 Conformance language

The key words `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.

- `MUST` / `MUST NOT`: mandatory architectural requirement.
- `SHOULD`: recommended requirement; deviations require explicit rationale.
- `MAY`: optional behavior explicitly allowed by the architecture.

## 0.5 Authority order

When documents differ, resolve in this order:

1. `docs/isa/*.md` for per-instruction semantics and constraints.
2. `include/pto/common/pto_instr.hpp` for public API surface and overload shape.
3. This manual for architecture layering, contracts, and conformance policy.

## 0.6 Reading order

1. `manual/01-overview.md`
2. `manual/02-machine-model.md`
3. `manual/03-state-and-types.md`
4. `manual/04-tiles-and-globaltensor.md`
5. `manual/05-synchronization.md`
6. `manual/06-assembly.md`
7. `manual/07-instructions.md`
8. `manual/08-programming.md`
9. `manual/09-virtual-isa-and-ir.md`
10. `manual/10-bytecode-and-toolchain.md`
11. `manual/11-memory-ordering-and-consistency.md`
12. `manual/12-backend-profiles-and-conformance.md`
13. `manual/appendix-a-glossary.md`
14. `manual/appendix-b-instruction-contract-template.md`
15. `manual/appendix-c-diagnostics-taxonomy.md`
16. `manual/appendix-d-instruction-family-matrix.md`
