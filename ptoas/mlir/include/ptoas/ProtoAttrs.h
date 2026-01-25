#pragma once

namespace ptoas {

// Shared prototype attributes used across MLIR passes and codegen.
inline constexpr const char *kStageAttr = "ptoas.stage";
inline constexpr const char *kPreambleStage = "__preamble__";

// Stage-specific tile address map attached to `pto.alloc_tile`.
// Type: DictionaryAttr<string stage, StringAttr addrLiteral>.
inline constexpr const char *kTileAddrMapAttr = "ptoas.tile_addrs";

// Memory model passed via CLI (e.g. MEMORY_BASE / REGISTER_BASE).
inline constexpr const char *kMemoryModelAttr = "ptoas.memory_model";

} // namespace ptoas
