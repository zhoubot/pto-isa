#pragma once

#include <string>

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace ptoas {

// Parses PTO-AS (DPS) text into an MLIR module containing unregistered `pto.*`
// operations with string attributes:
//
// - `pto.arg`:     {name = "...", type = "..."}
// - `pto.const`:   {name = "...", value = "...", type = "..."}
// - `pto.<instr>`: {operands = ["%a", "%b[%i,%j]", ...], attrs = "{...}", typesig = "(...)"}
//
// The module is suitable for running MLIR passes over op order, then emitting CCE.
mlir::ModuleOp parsePTOASFile(const std::string &path, mlir::MLIRContext &ctx, std::string &errorOut);

} // namespace ptoas

