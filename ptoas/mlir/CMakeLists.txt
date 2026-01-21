cmake_minimum_required(VERSION 3.20)
project(ptoas_mlir LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(SYSTEM
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)
link_directories(${LLVM_LIBRARY_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(ptoas_core
  lib/PTOASFrontend.cpp
  lib/CCEmitter.cpp
  lib/BishengDriver.cpp
)
llvm_update_compile_flags(ptoas_core)
target_include_directories(ptoas_core PUBLIC include)
target_link_libraries(ptoas_core PUBLIC
  MLIRIR
  MLIRSupport
  LLVMSupport
)

add_library(ptoas_passes
  lib/InsertEventsPass.cpp
)
llvm_update_compile_flags(ptoas_passes)
target_include_directories(ptoas_passes PUBLIC include)
target_link_libraries(ptoas_passes PUBLIC
  ptoas_core
  MLIRPass
  MLIRIR
  MLIRSupport
  LLVMSupport
)

add_executable(ptoas tools/ptoas_main.cpp)
set_target_properties(ptoas PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
llvm_update_compile_flags(ptoas)
target_include_directories(ptoas PRIVATE include)
target_link_libraries(ptoas PRIVATE
  ptoas_core
  ptoas_passes
  MLIRIR
  MLIRPass
  MLIRSupport
  LLVMSupport
)
