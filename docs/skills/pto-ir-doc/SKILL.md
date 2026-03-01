---
name: pto-ir-doc
description: Draft and maintain PTO IR documentation with architecture-level precision, including operand typing, SSA contracts, legality rules, and lowering boundaries.
---

# PTO IR Documentation

Use this skill when documenting PTO intermediate representation design and contracts.

## Workflow

1. Establish IR scope and ownership boundaries:
   - virtual ISA layer vs backend-specific lowering.
   - textual syntax vs in-memory IR model.
2. Define IR entities explicitly:
   - op names and categories.
   - operand/result types.
   - attributes and default semantics.
   - side effects and ordering semantics.
3. Specify legality and verifier rules:
   - type/shape compatibility.
   - location constraints (Vec/Mat/Left/Right/Acc/Bias/Scale).
   - valid-region semantics and undefined behavior boundaries.
4. Provide canonical examples:
   - minimal legal example.
   - common optimization pattern.
   - illegal example with reason.
5. Keep implementation traceability:
   - cross-link to `include/pto/common/pto_instr.hpp` and corresponding `docs/isa/*.md`.
   - mark implementation-defined behavior explicitly.

## Quality Rules

- Use RFC-2119 terms (MUST/SHOULD/MAY) for normative requirements.
- Do not leave implicit operand meaning; define all fields.
- Separate architecture contract from target-specific notes.
- Keep examples deterministic and copy-pastable.
