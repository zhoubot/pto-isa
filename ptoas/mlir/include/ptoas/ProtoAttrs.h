#pragma once

namespace ptoas {

// Shared prototype attributes used across MLIR passes and codegen.
inline constexpr const char *kStageAttr = "ptoas.stage";
inline constexpr const char *kPreambleStage = "__preamble__";

// Stage-specific tile address map attached to `pto.alloc_tile`.
// Type: DictionaryAttr<string stage, StringAttr addrLiteral>.
inline constexpr const char *kTileAddrMapAttr = "ptoas.tile_addrs";

} // namespace ptoas

