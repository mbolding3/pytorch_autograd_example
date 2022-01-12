#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
import cupy as cp
from math import gcd
from scipy.sparse import lil_matrix
from scipy.signal import resample_poly


def cp_to_torch(cp_output):
    return torch.as_tensor(cp.asnumpy(cp_output), device='cuda')


class FuncToyResample(Function):
    '''
    Naive (low performance) implementation of a polyphase resampler.

    1d only.
    '''
    @staticmethod
    def get_up_mat(length, up):
        out = lil_matrix(np.zeros((length * up, length)))
        for i in range(length):
            out[i * up, i] = 1
        return(out)

    @staticmethod
    def get_down_mat(length, down):
        down = get_up_mat(length, down)
        return(down.transpose())

    @staticmethod
    def forward(ctx, input, filter_coeffs, up, down):
        # input and filter_coeffs are tensors, up and down are ints
        input         = input.detach()
        filter_coeffs = filter_coeffs.detach()
        ctx.save_for_backward(input, filter_coeffs)
        input         = input.numpy()
        filter_coeffs = filter_coeffs.numpy()

        up_mat = get_up_mat(input.shape[0], up)
        down_mat = get_down_mat(input.shape[0], down)

        up = up_mat.dot(input)
        filtered = np.convolve(up, filter_coeffs, mode='valid')
        out = down_mat.dot(filtered)
        return torch.as_tensor(out, dtype=input.dtype)


class Resample(Module):
    def __init__(self):
        super(CusignalFFT, self).__init__()

    def forward(self, input):
        return FuncCusignalFFT.apply(input)


def sanity_main():
    x = np.random.rand(10)
    print(x)
    print(resample_poly(x, 1, 1))


def get_start_index(length):
    if length <= 2:
        return 0
    return (length - 1) // 2


def downsample_compare(repetitions = 1):
    for i in range(repetitions):
        x = np.random.rand(100)
        filt_len = np.random.randint(1,20)
        down = np.random.randint(1,10)
        filt = np.random.rand(filt_len)
        x_sp = resample_poly(x, 1, down, window=filt)
        x_np = np.convolve(x, filt)
        start = get_start_index(filt_len)
        if down == 1:
            x_np = x
        else:
            x_np = x_np[start::down]
        x_np = x_np[:len(x_sp)]
        same = np.allclose(x_np, x_sp)
        if not same:
            print(f"Naive and poly implementations agree? {same}")
            print(f"filter length: {filt_len}")
            print(f"down: {down}")
            print(f"x conv: {np.convolve(x, filt)[:15]}")
            print(f"x_np: {x_np[:5]}")
            print(f"x_sp: {x_sp[:5]}")


def upsample_compare(repetitions = 1):
    for i in range(repetitions):
        x_size = 100
        filt_size = np.random.randint(1,30)
        up = np.random.randint(1,10)
        x = np.random.rand(x_size)
        filt = np.random.rand(filt_size)
        x_np = np.zeros(up * x_size)
        x_np[::up] = up * x
        if (up > 1):
            start = get_start_index(filt_size)
            x_np = np.convolve(x_np, filt)
            x_np = x_np[start:]
        x_sp = resample_poly(x, up, 1, window=filt)
        if len(x_sp) > len(x_np):
            x_np = np.concatenate([x_np,
                                  np.zeros(len(x_sp)-len(x_np))])
        else:
            x_np = x_np[:len(x_sp)]
        same = np.allclose(x_np, x_sp)
        if not same:
            print(f"Naive and poly implementations agree? {same}")
            print(f"filter length: {filt_size}")
            print(f"up: {up}")
            print(f"x: {x}")
            print(f"x_np: {x_np}")
            print(f"x_sp: {x_sp}")


def polyphase_compare(repetitions = 1):
    for i in range(repetitions):
        x_size = np.random.randint(10, 100)
        filt_size = np.random.randint(1, x_size // 2)
        up = np.random.randint(1, 30)
        down = np.random.randint(1, 30)
        ud_gcd = gcd(up, down)
        up = up // ud_gcd
        down = down // ud_gcd
        x = np.random.rand(x_size)
        filt = np.random.rand(filt_size)
        x_np = np.zeros(up * x_size)
        x_np[::up] = up * x
        start = get_start_index(filt_size)
        if (up == 1 and down == 1):
            x_np = x
        elif (up > 1 and down == 1):
            # just up-sample
            x_np = np.convolve(x_np, filt)
            x_np = x_np[start:]
        elif (up == 1 and down > 1):
            # just down-sample
            x_np = np.convolve(x_np, filt)
            x_np = x_np[start::down]
        else:
            # non-trivial up and down
            x_np = np.convolve(x_np, filt)
            x_np = x_np[start::down]
        x_sp = resample_poly(x, up, down, window=filt)
        x_np = x_np[:len(x_sp)]
        same = np.allclose(x_np, x_sp)
        if not same:
            print(f"Naive and poly implementations agree? {same}")
            print(f"up: {up}, down: {down}, filt_size: {filt_size}")
            print(f"x_np: {x_np}")
            print(f"x_sp: {x_sp}")


if __name__ == '__main__':
    #downsample_compare(1000)
    #upsample_compare(1000)
    polyphase_compare(1000)
