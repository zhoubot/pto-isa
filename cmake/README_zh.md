# CMake 构建

本目录放仓库用到的 CMake 辅助模块，主要覆盖打包流程与第三方依赖集成。

## 目录内容

- 与打包相关的 CMake 逻辑（`package` 目标）
- 内置的 `makeself` 安装包生成
- `cmake/third_party/` 下的第三方依赖下载/集成脚本

## 关键文件

- `cmake/package.cmake`：由顶层 `CMakeLists.txt` 引入的打包入口函数
- `cmake/makeself_built_in.cmake`：内置 `makeself` 打包逻辑
- `cmake/third_party/`：第三方依赖辅助脚本

## 入口

- 顶层 `CMakeLists.txt` 会引入 `cmake/package.cmake` 并调用相关打包辅助函数
- `build.sh --pkg` 会触发仓库的打包流程
