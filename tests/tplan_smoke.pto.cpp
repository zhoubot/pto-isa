// Compile-only smoke test for TPLAN(...).
// Must be built with the PTO/AICORE toolchain (not a host compiler).

#include <pto/common/pto_instr.hpp>

namespace test {

using V0 = pto::Tile<pto::TileType::Vec, int8_t, 1, 32>;
using V1 = pto::Tile<pto::TileType::Vec, int8_t, 1, 32>;

__aicore__ inline void tplan_ok()
{
    V0 v0;
    V1 v1;

    // User-provided addresses, checked as a whole plan:
    TPLAN(
        TASSIGN<0x0000>(v0),
        TASSIGN<0x2000>(v1)
    );
}

} // namespace test
