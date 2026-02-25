/**
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef TQUANT_HPP
#define TQUANT_HPP

#include <pto/common/constants.hpp>
#include <pto/common/utils.hpp>
#include <pto/npu/kirin9030/common.hpp>
#include <pto/npu/kirin9030/utils.hpp>

namespace pto {
template <typename TileDataSrc, typename TileDataExp, typename TileDataOut, typename TileDataMax, int mode>
PTO_INTERNAL void TQUANT_IMPL(TileDataSrc &src, TileDataExp &exp, TileDataOut &dst, TileDataMax &max,
                              TileDataSrc &scaling)
{
    using T = typename TileDataSrc::DType;
    static_assert(sizeof(T) == 0, "Fix: Unsupports the instruction: TQUANT");
}
} // namespace pto
#endif // TQUANT_HPP