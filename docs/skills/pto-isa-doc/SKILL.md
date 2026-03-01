---
name: pto-isa-doc
description: Write and review PTO ISA specifications and documentation (virtual ISA, IR, bytecode, assembler) with clear architectural state, memory/exception model, and testable normative language.
---

# PTO ISA Documentation

Use this skill when you need to author or review PTO ISA documentation in a way that is precise enough for compiler/runtime/driver implementers, and readable enough for users.

This skill complements: `pto-virtual-isa-doc`, `pto-ir-doc`, `pto-bytecode-doc`, and general writing/style skills like `technical-writing` and `documentation-standards`.

## Workflow

### Pick the spec layer

- **Virtual ISA manual**: abstract machine contract, independent of backend microarchitecture.
- **IR spec**: typing/SSA legality and lowering boundaries.
- **Bytecode/assembly spec**: textual syntax, module format, validation rules, diagnostics.

If the user request mixes layers, split the deliverable and state which rules belong to which layer.

### Establish the “programmer’s model”

Define these up front and reuse terms consistently:

- **Execution model**: threads/waves/tiles, divergence, reconvergence rules (if any).
- **Architectural state**: registers, predicates/masks, special registers, memory spaces.
- **Type system**: scalar/vector/tensor shapes, lane semantics, rounding/saturation, NaN rules.
- **Memory model**: ordering, visibility scopes, atomics, fences, and “undefined vs implementation-defined”.
- **Exception/diagnostic model**: traps vs validation errors vs “poison/UB” style behavior.
- **Versioning**: feature gates, deprecation policy, and compatibility guarantees.

### Specify instructions as contracts (not prose)

For each instruction (or instruction family), include at least:

- **Name and purpose** (one sentence).
- **Syntax** (assembly/IR form) and **operand roles**.
- **Type/shape rules** (including implicit conversions and legality constraints).
- **Semantics** (pseudocode or equivalent, including side effects).
- **Memory effects** (read/write sets, ordering, address space).
- **Failure modes** (trap/exception, or validation rule; avoid “unspecified” without saying what varies).
- **Notes** (performance hints only if they don’t change correctness).
- **Example(s)** with expected behavior.

Use normative language (RFC 2119 keywords like MUST/SHOULD/MUST NOT) for requirements. Avoid hedging words (“usually”, “typically”) in normative statements.

### Keep docs source-aligned

In this repo, validate claims against (at minimum):

- `include/pto/common/pto_instr.hpp`
- `docs/PTOISA.md` and `docs/PTO-Virtual-ISA-Manual.md`
- `docs/bytecode/` (when documenting PTOAS / bytecode)

Prefer updating docs to match code (or vice versa), and call out any intentional divergences explicitly.

## Output templates

### Instruction template (Markdown)

Use this skeleton for a new instruction page/section:

- **Summary**: what it does, in one sentence.
- **Form**: syntax (and variants / suffixes).
- **Operands**: table of name, kind, type, constraints, description.
- **Type rules**: legality, inference, promotions, shape constraints.
- **Semantics**: pseudocode; define helper functions once and reuse.
- **Memory model**: ordering/scope/atomicity if applicable.
- **Exceptions / validation**: exact conditions and diagnostic text guidelines.
- **Notes**: implementation-defined behavior, portability notes.
- **Examples**: at least one minimal example and one edge case.

### Review checklist

- Every requirement is testable.
- Terminology is defined once and reused.
- “Undefined”, “unspecified”, and “implementation-defined” are used intentionally and precisely.
- No silent type conversions or hidden state changes without being specified.
- Examples compile (or are clearly marked as pseudocode) and match the described semantics.

## Copy/paste prompt (for writing a spec section)

You are writing a normative ISA specification for PTO.

Task: write/update the spec for: {feature_or_instruction_family}.

Constraints:
- Target audience: compiler/runtime implementers first; end-user readability second.
- Use RFC 2119 keywords for requirements.
- Separate: (1) legality/type rules, (2) dynamic semantics, (3) diagnostics/exceptions, (4) portability notes.
- If behavior is implementation-defined, name the variation points explicitly.

Inputs you must consult (if present in the workspace):
- Code: {relevant_source_files_or_search_terms}
- Existing docs: {relevant_doc_files}

Output format:
- Markdown with headings (no manual heading numbering).
- Include at least one worked example and one edge-case example.
- End with a short “Open questions / TODO” list if anything is ambiguous or missing in sources.
