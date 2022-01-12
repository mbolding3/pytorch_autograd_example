#!/usr/bin/env python3


import torch
from torch import flip
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


def get_start_index(length):
    if length <= 2:
        return 0
    return (length - 1) // 2


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
    def forward(ctx, x, filter_coeffs, up, down):
        x             = x.detach()
        filter_coeffs = filter_coeffs.detach()
        ctx.save_for_backward(x, filter_coeffs)

        x_size = x.shape[0]
        filt_size = filter_coeffs.shape[0]
        ud_gcd = gcd(up, down)
        up = up // ud_gcd
        down = down // ud_gcd

        x_up = torch.zeros(up * x_size, device = x.device.type)
        # This is probably terrible for device data.
        x_up[::up] = up * x
        start = get_start_index(filt_size)
        if (up == 1 and down == 1):
            x_out = x_up
        elif (up == 1 and down > 1):
            # just down-sample
            x_out = torch.conv1d(x_up.reshape(1, 1, x_up.shape[0]),
                                 flip(filter_coeffs, [0]).\
                                 reshape(1, 1, filt_size),
                                 padding = filt_size - 1)
            np_conv = np.convolve(x_up.numpy(), filter_coeffs.numpy())
            x_out = x_out.reshape(x_out.shape[-1])[start::down]
        elif (up > 1 and down == 1):
            # just up-sample
            x_out = torch.conv1d(x_up.reshape(1, 1, x_up.shape[0]),
                                 flip(filter_coeffs, [0]).\
                                 reshape(1, 1, filt_size),
                                 padding = filt_size - 1)
            np_conv = np.convolve(x_up.numpy(), filter_coeffs.numpy())
            x_out = x_out.reshape(x_out.shape[-1])[start:]
        else:
            # non-trivial up and down
            x_out = torch.conv1d(x_up.reshape(1, 1, x_up.shape[0]),
                                 flip(filter_coeffs, [0]).\
                                 reshape(1, 1, filt_size),
                                 padding = filt_size - 1)
            np_conv = np.convolve(x_up.numpy(), filter_coeffs.numpy())
            x_out = x_out.reshape(x_out.shape[-1])[start::down]
        out_len = x_size * up
        out_len = out_len // down + bool(out_len % down)
        x_out = x_out[:out_len]
        return(x_out)


class Resample(Module):
    def __init__(self):
        super(Resample, self).__init__()

    def forward(self, x, up, down, filter_coeffs):
        return FuncToyResample.apply(x, filter_coeffs, up, down)


def scipy_check_main(repetitions = 1):
    module = Resample()
    for i in range(repetitions):
        x_size = np.random.randint(30, 100)
        filter_size = np.random.randint(5, 20)
        x = torch.randn(x_size)
        f = torch.randn(filter_size)
        up = np.random.randint(1, 20)
        down = np.random.randint(1, 20)
        scipy_resample = resample_poly(x, up, down, window = f.numpy())
        our_resample = module.forward(x, up, down, f)
        if not np.allclose(scipy_resample, our_resample, atol=1e-4):
            print(f"up: {up}, down: {down}")
            print(f"scipy result: {scipy_resample[:10]}")
            print(f"our result: {our_resample[:10]}")


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
        out_len = x_size * up
        out_len = out_len // down + bool(out_len % down)
        x_np = x_np[:out_len]
        same = np.allclose(x_np, x_sp)
        if not same:
            print(f"Naive and poly implementations agree? {same}")
            print(f"up: {up}, down: {down}, filt_size: {filt_size}")
            print(f"x_np: {x_np}")
            print(f"x_sp: {x_sp}")


if __name__ == '__main__':
    #downsample_compare(1000)
    #upsample_compare(1000)
    #polyphase_compare(1000)
    scipy_check_main(1000)
