#!/usr/bin/python3
# coding=utf-8
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

import op_extension


class TestPtoGemmBasic(TestCase):

    def test_pto_gemm_basic(self):
        m, k, n = 512, 2048, 1536
        a = torch.rand((m, k), device='cpu', dtype=torch.float16)
        b = torch.rand((k, n), device='cpu', dtype=torch.float16)
        b_dn = b.t().contiguous()

        a_npu = a.npu()
        b_dn_npu = b_dn.npu()
        out = torch.ops.npu.pto_gemm_basic(a_npu, b_dn_npu)

        ref = torch.matmul(a.float(), b.float())
        self.assertRtolEqual(out.cpu(), ref)


if __name__ == "__main__":
    run_tests()

