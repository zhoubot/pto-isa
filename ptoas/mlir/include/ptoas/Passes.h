#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace ptoas {

std::unique_ptr<mlir::Pass> createInsertEventsPass();
std::unique_ptr<mlir::Pass> createAssignTileAddressesPass();

} // namespace ptoas
