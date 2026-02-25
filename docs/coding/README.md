# docs/coding/

This directory describes the **PTO Tile Lib programming model as seen from C++** (Tiles, GlobalTensor, events, scalar parameters) and provides guidance for extending the library.

If you are looking for the *ISA reference*, start from `docs/isa/README.md`.

## Documents

- High-level model and PTO-Auto/PTO-Manual: `docs/coding/ProgrammingModel.md`
- Hands-on tutorial (write your first kernels): `docs/coding/tutorial.md`
- More tutorial examples: `docs/coding/tutorials/README.md`
- Debugging and assertion lookup: `docs/coding/debug.md`
- Tile abstraction and layout/valid-region rules: `docs/coding/Tile.md`
- Global memory tensors (shape/stride/layout): `docs/coding/GlobalTensor.md`
- Events and synchronization model: `docs/coding/Event.md`
- Scalar values, type mnemonics, and enums: `docs/coding/Scalar.md`

## Related

- PTO abstract machine model: `docs/machine/README.md`
- ISA reference: `docs/isa/README.md`

## Related Components

| Component | Description | Path |
|-----------|-------------|------|
| **PTO ISA** | Virtual instruction set architecture | [`include/pto/`](include/pto/) |
| **PTO-AS** | Assembly language specification | [`docs/grammar/PTO-AS.md`](docs/grammar/PTO-AS.md) |
| **pyPTO** | Python frontend for PTO | [`docs/pyPTO/`](docs/pyPTO/) |
| **ptoas** | PTO assembler and MLIR dialect | [`ptoas/`](ptoas/) |
| **TileLang Ascend** | High-level framework integration | [External: tilelang-ascend](https://github.com/tile-ai/tilelang-ascend/) |
