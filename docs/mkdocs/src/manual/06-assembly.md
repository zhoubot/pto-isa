# 6. PTO assembly (PTO-AS)

## 6.1 Scope

This chapter defines the Virtual ISA contract of PTO-AS as the textual form of PTO programs.
The normative grammar remains:

- `docs/grammar/PTO-AS.md`
- `docs/grammar/PTO-AS.bnf`

## 6.2 Core form

PTO-AS uses an instruction-centric SSA-like textual form.
A typical statement shape is:

```text
%dst = tadd %src0, %src1 : (!pto.tile<...>, !pto.tile<...>) -> !pto.tile<...>;
```

The textual form SHOULD remain deterministic under round-trip tooling.

## 6.3 Operand classes

PTO-AS operands include:

- tile operands
- memory/global operands
- scalar/immediate operands
- event/dependency operands (where applicable)
- attributes/modifiers expressed by dictionary form

Each instruction family MUST define required operand classes and positional constraints.

## 6.4 Attribute and modifier contract

Attributes MUST define:

- name and type
- allowed value domain
- default value policy (if any)
- semantic impact
- diagnostics behavior for invalid values

## 6.5 Structural validity rules

A structurally valid PTO-AS program MUST satisfy:

- operand/result arity consistency
- type-class compatibility per operation contract
- required attribute presence
- parseable and schema-valid statement forms

## 6.6 Diagnostics contract

PTO-AS diagnostics MUST be:

- location-aware for parse and structural errors
- deterministic for equivalent inputs
- actionable with expected-vs-actual constraints

## 6.7 Compatibility and evolution

PTO-AS evolution SHOULD be additive.
Breaking textual-syntax changes MUST be versioned and accompanied by migration guidance.
Toolchains MUST reject unsupported syntax with deterministic diagnostics.
