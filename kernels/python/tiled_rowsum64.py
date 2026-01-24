from __future__ import annotations

from pto_as import PTO


def tiled_rowsum64():
    # Row reduction on a (64,64) tensor using 4 tiles per row block.
    #
    # NOTE:
    # - `trowsum` produces a (16,1) tile (typically ColMajor on A2/A3).
    # - Elementwise `tadd/tsub` do not support (16,1) ColMajor tiles on A2/A3, so
    #   we accumulate by expanding row sums into a (16,16) tile via `trowexpandadd`.
    #
    # Output: y has shape (64,16). Each row is the row-sum replicated across 16 columns.
    pto = PTO("tiled_rowsum64")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(64, 64), role="in")
    y = pto.tensor(dtype="f32", shape=(64, 16), role="out")

    tx = pto.vec(dtype="f32", shape=(16, 16))
    tmp = pto.vec(dtype="f32", shape=(16, 16))
    row_sum = pto.vec(dtype="f32", shape=(16, 1), blayout="ColMajor")
    bcast = pto.vec(dtype="f32", shape=(16, 16))
    acc = pto.vec(dtype="f32", shape=(16, 16))

    for r in range(0, 64, 16):
        # Init accumulator to 0 using the first tile.
        tx = pto.load(x, r, 0)
        acc = pto.tsub(tx, tx)

        # Add row sums from each (16,16) tile in the row block.
        for c in range(0, 64, 16):
            tx = pto.load(x, r, c)
            row_sum = pto.trowsum(tx, tmp)
            bcast = pto.trowexpand(row_sum)
            acc = pto.tadd(acc, bcast)

        pto.store(y, r, 0, acc)

    pto.epilogue()
    return pto.program()
