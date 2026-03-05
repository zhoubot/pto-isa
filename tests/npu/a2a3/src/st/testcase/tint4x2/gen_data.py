#!/usr/bin/python3
# coding=utf-8
import os
import numpy as np

np.random.seed(123)

def gen(case_dir, nbytes):
    os.makedirs(case_dir, exist_ok=True)
    x = np.random.randint(0, 256, size=(nbytes,), dtype=np.uint8)
    x.tofile(os.path.join(case_dir, 'input.bin'))
    x.tofile(os.path.join(case_dir, 'golden.bin'))

if __name__ == '__main__':
    # Keep same workflow as other STs: per-case dir with input/golden.
    gen('TINT4X2Test.case_copy_64x64', 64 * 64)
    gen('TINT4X2Test.case_copy_32x128', 32 * 128)
    # Valid cols = 95, but capacity cols = 96; we still provide full tile payload.
    gen('TINT4X2Test.case_copy_32x96_v32x95', 32 * 96)
