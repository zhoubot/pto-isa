---
name: pto-bytecode-doc
description: Document PTO-AS and bytecode/toolchain behavior, including textual assembly mapping, bytecode module contracts, validation, and diagnostics.
---

# PTO Bytecode Documentation

Use this skill when documenting PTO-AS and bytecode interchange/toolchain behavior.

## Workflow

1. Define representation layers:
   - PTO virtual ISA semantics.
   - PTO-AS textual form.
   - MLIR-style bytecode interchange form.
2. Specify module-level contract:
   - symbol/function layout.
   - SSA value naming model.
   - type and attribute preservation.
3. Define validation stages:
   - syntax parsing.
   - structural verification.
   - optional target legality checks.
4. Define diagnostics requirements:
   - line/column anchoring for syntax errors.
   - deterministic verifier error wording.
5. Document compatibility and evolution policy:
   - versioning field requirements.
   - forward/backward compatibility expectations.

## Quality Rules

- Keep bytecode contract practical and implementable.
- Mark unsupported/undefined sections explicitly.
- Provide round-trip examples (text -> bytecode -> text).
