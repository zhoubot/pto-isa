#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace pto;

TEST(TStoreAtomicAdd, ND_float_acc_add_twice)
{
    using T = float;

    using GShape = Shape<1, 1, 1, 2, 128>;
    using GStride = BaseShape2D<T, 2, 128, Layout::ND>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;

    alignas(64) T gm[2 * 128];
    for (int i = 0; i < 2 * 128; ++i) {
        gm[i] = 1.0f;
    }

    constexpr int Rows = 16;
    constexpr int Cols = 128;
    using AccTile = Tile<TileType::Acc, T, Rows, Cols, BLayout::ColMajor, -1, -1, SLayout::RowMajor, 1024>;

    AccTile t(2, 128);

    auto &td = t.data();
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 128; ++c) {
            td[GetTileElementOffset<AccTile>(r, c)] = 2.0f;
        }
    }

    GT g(gm);

    TSTORE<AccTile, GT, AtomicType::AtomicAdd>(g, t);
    TSTORE<AccTile, GT, AtomicType::AtomicAdd>(g, t);

    for (int i = 0; i < 2 * 128; ++i) {
        EXPECT_FLOAT_EQ(gm[i], 5.0f);
    }
}
