# 9. Virtual ISA and IR

## 9.1 Scope and normative terms

This chapter defines the contract between PTO Virtual ISA semantics and PTO IR/lowering pipelines.
The terms `MUST`, `MUST NOT`, `SHOULD`, and `MAY` are normative.

## 9.2 Layering model

PTO uses a three-layer contract:

1. **Virtual ISA layer**: architecture-visible semantics.
2. **IR layer**: structured typed representation for verification and transformation.
3. **Backend lowering layer**: target-specific legalization and code generation.

Backend specialization MUST preserve Virtual ISA-observable behavior.

## 9.3 IR object model

A conforming PTO IR model SHOULD define:

- module and symbol contracts
- function/block structure and ordering
- SSA value topology
- operation schema (name, operands, results, attributes, effects)
- explicit synchronization and memory effects

## 9.4 Verifier boundary

Verification is split into two levels:

1. **Structural verifier (IR level)**
- MUST validate operation schema, arity, type classes, and required attributes.
- MUST be target-independent.

2. **Target legality verifier (backend level)**
- MUST validate dtype/layout/location/shape tuples for selected backend profile.
- MUST produce deterministic diagnostics for unsupported tuples.

## 9.5 Lowering invariants

Lowering MUST preserve:

- valid-region semantics
- explicit ordering dependencies (`event`, `TSYNC`, memory-ordering points)
- operation meaning within architecture-defined domains

Lowering MUST NOT silently reinterpret implementation-defined behavior as architecture-defined behavior.

## 9.6 Source alignment rules

IR contracts MUST stay synchronized with:

- `docs/isa/*.md` for semantic intent
- `include/pto/common/pto_instr.hpp` for API-level shape
- `docs/grammar/PTO-AS.md` for textual assembly-facing forms

## 9.7 Compatibility policy

- Additive IR changes SHOULD be preferred.
- Breaking IR contract changes MUST include versioning and migration notes.
- Unknown required fields MUST fail verification.
- Deprecated constructs SHOULD remain parseable for at least one compatibility window.

## 9.8 Diagnostics requirements

IR/verifier diagnostics MUST include:

- operation identifier and location context
- expected vs actual contract dimensions
- deterministic error class suitable for CI regression

## 9.9 Minimum conformance scenarios

Conformance validation SHOULD include:

- legal and illegal structural verifier tests
- backend legality pass/fail matrix by profile
- round-trip checks through PTO-AS and bytecode forms
- differential checks against per-instruction semantics
