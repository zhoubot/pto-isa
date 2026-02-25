cmake_minimum_required(VERSION 3.20)
project(ptobc LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

# Prefer the in-tree PTOAS install by default, but allow overriding via:
#   -DPTOAS_ROOT=/path/to/PTOAS/install
set(PTOAS_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../PTOAS/install" CACHE PATH "Path to PTOAS install prefix")
if (EXISTS "${PTOAS_ROOT}/lib/cmake/PTOAS/PTOASConfig.cmake")
  list(APPEND CMAKE_PREFIX_PATH "${PTOAS_ROOT}")
endif()
find_package(PTOAS CONFIG REQUIRED)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using PTOASConfig.cmake in: ${PTOAS_DIR}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_executable(ptobc
  src/main.cpp
  src/leb128.cpp
  src/ptobc_format.cpp
  src/mlir_helpers.cpp
  src/mlir_encode.cpp
  src/ptobc_decode_print.cpp
)

target_include_directories(ptobc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/generated
  ${PTOAS_INCLUDE_DIRS}
)

# MLIR deps
llvm_map_components_to_libnames(LLVM_LIBS Support)

target_link_libraries(ptobc PRIVATE
  ${LLVM_LIBS}
  PTOAS::PTOIR
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRFuncDialect
  MLIRArithDialect
  MLIRSCFDialect
)

# NOTE: We only link PTOAS::PTOIR (dialect + IR). We intentionally do NOT link
# transform libraries. The encoder still stores types/attrs as asm strings for
# lossless round-trip, but now we can parse PTODSL-produced `.pto` without
# requiring users to quote every pto op.
