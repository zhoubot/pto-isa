# PTO Bytecode (PTO-BC)

This directory tracks the design of a **binary encoding** for PTO programs.

## What is being encoded?

Today, `PTOAS` consumes textual `.pto` files that are MLIR modules using:

- `pto` dialect (tile-level ops, tensor views, events)
- `arith` dialect (constants, integer/index arithmetic)
- `scf` dialect (structured control flow: `for`, `if`)

Examples are generated from `PTODSL` and stored in [`samples/`](samples/).

## Goal

Design a compact, forward-compatible **bytecode format** for `.pto` programs that:

- is independent of MLIR's internal serialization
- can be decoded without linking full MLIR (optional goal)
- preserves enough typing/structure to reconstitute the same program
- leaves clear extension space for new PTO ops / attrs / types

See: [`pto-bc.md`](pto-bc.md)
