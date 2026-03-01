---
name: pto-virtual-isa-doc
description: Produce professional PTO virtual-ISA documentation covering abstract machine contracts, instruction families, portability rules, and backend specialization boundaries.
---

# PTO Virtual ISA Documentation

Use this skill when authoring high-level PTO ISA guidance for compiler, runtime, and kernel developers.

## Workflow

1. Define virtual ISA contract:
   - stable semantics independent of hardware generation.
   - explicit boundary between architecture and implementation-defined behavior.
2. Organize instruction families and contracts:
   - data movement, elementwise, reduce/expand, matmul/gemv, gather/scatter, sync/config.
3. Specify portability model:
   - what is guaranteed across A2/A3/A5/CPU.
   - what depends on backend constraints.
4. Capture programming models:
   - Auto vs Manual.
   - SPMD/MPMD mapping.
   - synchronization and dependency model.
5. Keep source alignment:
   - verify against `include/pto/common/pto_instr.hpp` and `docs/isa/` pages.

## Quality Rules

- Requirements must be testable.
- Term definitions must be stable and reused consistently.
- Avoid ambiguous phrasing like "usually" or "typically" in normative statements.
