#include <pto/pto-inst.hpp>
#include <gtest/gtest.h>

using namespace pto;

TEST(TStoreAtomicAdd, ND_float_add_twice) {
    using T = float;
    constexpr int R = 4;
    constexpr int C = 8;

    using TileT = Tile<TileType::Vec, T, R, C, BLayout::RowMajor>;
    using GShape = Shape<1,1,1,R,C>;
    using GStride = BaseShape2D<T, R, C, Layout::ND>;
    using GT = GlobalTensor<T, GShape, GStride, Layout::ND>;

    alignas(64) T gm[C*R];
    for (int i=0;i<R*C;i++) gm[i] = 1.0f;

    TileT t;
    // fill tile with 2
    auto &td = t.data();
    for (int r=0;r<R;r++) for (int c=0;c<C;c++) td[r*C+c] = 2.0f;

    GT g(gm);

    // atomic add twice
    TSTORE<TileT, GT, AtomicType::AtomicAdd>(g, t);
    TSTORE<TileT, GT, AtomicType::AtomicAdd>(g, t);

    for (int i=0;i<R*C;i++) {
        EXPECT_FLOAT_EQ(gm[i], 1.0f + 2.0f + 2.0f);
    }
}
