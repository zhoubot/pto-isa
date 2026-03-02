# cmake/

This directory contains CMake helper modules used by the repository, mainly for packaging and third-party dependency integration.

## What’s Inside

- Packaging-related CMake logic (the `package` target)
- Built-in `makeself` installer generation
- Third-party dependency download/integration tests/scripts under `cmake/third_party/`

## Key Files

- `cmake/package.cmake`: Packaging entry functions included by the top-level `CMakeLists.txt`
- `cmake/makeself_built_in.cmake`: Built-in `makeself` packaging logic
- `cmake/third_party/`: Third-party dependency helper tests/scripts

## Entry Points

- Top-level `CMakeLists.txt` includes `cmake/package.cmake` and invokes the packaging helpers
- `build.sh --pkg` triggers the repository packaging flow
