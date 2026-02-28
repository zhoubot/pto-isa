from __future__ import annotations

from pto_as import PTO


def spmd_tiled_add_18x19():
    """
    Irregular-shape regression:
      - tensor shape: 18x19 (not multiples of 16)
      - tile shape:   16x16
      - 2x2 tiles total (block_dim = 4)

    This exercises:
      - dynamic tile valid masks (valid_row/valid_col)
      - indexed tload/tstore with partial tiles
    """
    pto = PTO("spmd_tiled_add_18x19")
    pto.prologue()

    x = pto.tensor(dtype="f32", shape=(18, 19), role="in")
    y = pto.tensor(dtype="f32", shape=(18, 19), role="in")
    z = pto.tensor(dtype="f32", shape=(18, 19), role="out")

    bid = pto.get_block_idx()

    tiles_c = 2  # ceil_div(19, 16)
    tile_c = bid % tiles_c
    tile_r = bid // tiles_c

    r0 = tile_r * 16
    c0 = tile_c * 16

    rem_r = 18 - r0
    rem_c = 19 - c0
    vr = pto.imin(16, rem_r)
    vc = pto.imin(16, rem_c)

    tx = pto.vec(dtype="f32", shape=(16, 16), valid_row=vr, valid_col=vc)
    ty = pto.vec(dtype="f32", shape=(16, 16), valid_row=vr, valid_col=vc)
    tz = pto.vec(dtype="f32", shape=(16, 16), valid_row=vr, valid_col=vc)

    tx = pto.load(x, r0, c0)
    ty = pto.load(y, r0, c0)
    tz = pto.add(tx, ty)
    pto.store(z, r0, c0, tz)

    pto.epilogue()
    return pto.program()

