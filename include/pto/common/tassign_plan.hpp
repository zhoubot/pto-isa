#ifndef PTO_COMMON_TASSIGN_PLAN_HPP
#define PTO_COMMON_TASSIGN_PLAN_HPP

#include <cstddef>
#include <type_traits>

#include <pto/common/buffer_limits.hpp>
#include <pto/common/pto_tile.hpp>

namespace pto {

// Compile-time address wrapper for TASSIGN/TPLAN.
// Prefer user-facing macro TADDR(x).
template <std::size_t Addr>
using TAddr = std::integral_constant<std::size_t, Addr>;

namespace detail {

// -----------------------
// Tile byte-size helpers
// -----------------------

template <typename T, bool IsConv = is_conv_tile_v<T>>
struct TileBytes;

template <typename T>
struct TileBytes<T, false> {
    static constexpr std::size_t value =
        static_cast<std::size_t>(T::Rows) * static_cast<std::size_t>(T::Cols) * sizeof(typename T::DType);
};

template <typename T>
struct TileBytes<T, true> {
    static constexpr std::size_t value = static_cast<std::size_t>(T::bufferSize) * sizeof(typename T::DType);
};

// -----------------------
// Buffer traits by TileType::Loc
// -----------------------

template <TileType Loc>
struct BufferTraits;

template <>
struct BufferTraits<TileType::Vec> {
    static constexpr std::size_t cap = PTO_UBUF_SIZE_BYTES;
    static constexpr std::size_t align = PTO_UBUF_ALIGN_BYTES;
};

template <>
struct BufferTraits<TileType::Bias> : BufferTraits<TileType::Vec> {
};

template <>
struct BufferTraits<TileType::Mat> {
    static constexpr std::size_t cap = PTO_CBUF_SIZE_BYTES;
    static constexpr std::size_t align = PTO_CBUF_ALIGN_BYTES;
};

template <>
struct BufferTraits<TileType::Left> {
    static constexpr std::size_t cap = PTO_L0A_SIZE_BYTES;
    static constexpr std::size_t align = PTO_L0A_ALIGN_BYTES;
};

template <>
struct BufferTraits<TileType::ScaleLeft> : BufferTraits<TileType::Left> {
};

template <>
struct BufferTraits<TileType::Right> {
    static constexpr std::size_t cap = PTO_L0B_SIZE_BYTES;
    static constexpr std::size_t align = PTO_L0B_ALIGN_BYTES;
};

template <>
struct BufferTraits<TileType::ScaleRight> : BufferTraits<TileType::Right> {
};

template <>
struct BufferTraits<TileType::Acc> {
    static constexpr std::size_t cap = PTO_L0C_SIZE_BYTES;
    static constexpr std::size_t align = PTO_L0C_ALIGN_BYTES;
};

template <>
struct BufferTraits<TileType::Scaling> {
    static constexpr std::size_t cap = PTO_FBUF_SIZE_BYTES;
    static constexpr std::size_t align = PTO_FBUF_ALIGN_BYTES;
};

// -----------------------
// Binding token & checks
// -----------------------

template <typename TileT, std::size_t Addr>
struct BindToken {
    using Tile = TileT;
    static constexpr TileType loc = TileT::Loc;
    static constexpr std::size_t addr = Addr;
    static constexpr std::size_t bytes = TileBytes<TileT>::value;
    static constexpr std::size_t cap = BufferTraits<loc>::cap;
    static constexpr std::size_t align = BufferTraits<loc>::align;

    static constexpr std::size_t end = addr + bytes;

    static constexpr bool in_range = end <= cap;
    static constexpr bool aligned = (align == 0) ? true : (addr % align == 0);
};

template <typename A, typename B>
struct Overlap {
    static constexpr bool value = !(A::end <= B::addr || B::end <= A::addr);
};

template <typename... Ts>
struct PlanCheck;

template <>
struct PlanCheck<> {
    static constexpr bool ok = true;
};

template <typename T0>
struct PlanCheck<T0> {
    static constexpr bool ok = T0::in_range && T0::aligned;
};

template <typename T0, typename... Rest>
struct PlanCheck<T0, Rest...> {
    static constexpr bool ok =
        (T0::in_range && T0::aligned) &&
        ((Rest::in_range && Rest::aligned) && ...) &&
        // no overlaps with T0
        ((!Overlap<T0, Rest>::value) && ...) &&
        // rest pairwise
        PlanCheck<Rest...>::ok;
};

template <typename... Binds>
struct Plan {
    static constexpr bool ok = PlanCheck<Binds...>::ok;
};

// Build a plan from bind tokens (values are unused; only types matter).
template <typename T>
struct is_bind_token : std::false_type {
};

template <typename TileT, std::size_t Addr>
struct is_bind_token<BindToken<TileT, Addr>> : std::true_type {
};

template <typename... Binds>
constexpr Plan<Binds...> MakePlan(Binds...)
{
    static_assert((is_bind_token<Binds>::value && ...),
                  "TPLAN expects bind tokens returned by TASSIGN(tile, TADDR(...)) or TASSIGN<ADDR>(tile).");
    static_assert(Plan<Binds...>::ok, "TPLAN: tile address plan invalid (alignment/range/overlap)." );
    return {};
}

} // namespace detail

// Public entry point: MakePlan(...) (argument must be bind tokens from TASSIGN with TAddr<>).
template <typename... Binds>
constexpr auto MakePlan(Binds... binds)
{
    return detail::MakePlan(binds...);
}

// User-facing address macro: forces compile-time constant address token.
// Usage: TASSIGN(t, TADDR(0x2000))
#define TADDR(x) (::pto::TAddr<(std::size_t)(x)>{})

// Preferred compile-time assign form:
//   TASSIGN<0x2000>(tile);
// (implemented as a function template overload in pto_instr.hpp)

} // namespace pto

// Group a set of TASSIGN(...) calls and enforce whole-program (whole-plan) checks.
//
// Usage:
//   TPLAN(
//     TASSIGN<0x0000>(t0),
//     TASSIGN<0x2000>(t1)
//   );
//
// Notes:
// - Each TASSIGN is executed (so it still does the normal binding).
// - The returned bind tokens are collected and checked as a set.
#define TPLAN(...) \
    do { \
        auto _pto_plan = ::pto::MakePlan(__VA_ARGS__); \
        (void)_pto_plan; \
    } while (0)

#endif // PTO_COMMON_TASSIGN_PLAN_HPP
