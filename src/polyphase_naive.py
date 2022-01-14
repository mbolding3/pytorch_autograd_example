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
    def forward(ctx, x, filter_coeffs, up, down):
        x             = x.detach()
        filter_coeffs = filter_coeffs.detach()
        up            = up.detach()
        down          = down.detach()
        x_size_og = torch.Tensor([x.shape[0]])
        f_og = torch.clone(filter_coeffs)
        up_og = torch.clone(up)
        down_og = torch.clone(down)


        x_size = x.shape[0]
        filt_size = filter_coeffs.shape[0]
        up = int(up[0])
        down = int(down[0])
        ud_gcd = gcd(up, down)
        up = up // ud_gcd
        down = down // ud_gcd

        if (up == 1 and down == 1):
            x_out = x
            inverse_size = torch.Tensor([x.shape[0]])
            out_len = torch.Tensor([0])
            x_up_og = None
        else:
            x_up = torch.zeros(up * x_size, device = x.device.type,
                               dtype = x.dtype)
            # This is probably terrible for device data.
            x_up[::up] = up * torch.clone(x)
            x_up_og = torch.clone(x_up)
            start = get_start_index(filt_size)
            x_conv = torch.conv1d(x_up.reshape(1, 1, x_up.shape[0]),
                                  flip(filter_coeffs, [0]).\
                                  reshape(1, 1, filt_size),
                                  padding = filt_size - 1)
            inverse_size = torch.Tensor([x_conv.shape[-1]])
            x_out = x_conv.reshape(x_conv.shape[-1])[start::down]
            out_len = x_size * up
            out_len = out_len // down + bool(out_len % down)
            x_out = x_out[:out_len]
            out_len = torch.Tensor([out_len])
        ctx.save_for_backward(x_size_og, f_og, up_og, down_og, inverse_size,
                              out_len, x_up_og)
        return(x_out)

    @staticmethod
    def backward(ctx, gradient):
        '''
        order of operations in the forward method:

        1) up-sample to size up * n, multiply by up
        2) do a zero-padded convolution
        3) down sample beginning at start index
        4) truncate output to size n * up / down + optional 1 if there is
           a remainder.

        It's the reshaping and striding that causes all the problems.
        Gradient_x of a padded convolution is a non-padded correlation - 
        that's the easy part, strangely.
        Gradient of a correlation with respect to the filter is again
        a correlation with respect to the same filter.
        '''
        gradient = gradient.detach()
        x_size, filter_coeffs, up, down, inverse_size, out_len, x_up \
        = ctx.saved_tensors

        x_size = int(x_size[0])
        gradient_size = gradient.shape[0]
        filt_size = filter_coeffs.shape[0]
        up = int(up[0])
        down = int(down[0])
        ud_gcd = gcd(up, down)
        up = up // ud_gcd
        down = down // ud_gcd
        start = get_start_index(filt_size)
        inverse_size = int(inverse_size)
        out_x_len = int(out_len)
        filter_coeffs = filter_coeffs.type(gradient.dtype)

        if (up != 1 or down != 1):
            # This is up-sampling by a factor down, and a bunch of
            # correcting the weird array truncations in forward.
            tmp = torch.zeros(out_x_len)
            tmp[:gradient.shape[0]] = gradient
            gradient = tmp          
            gradient_up = torch.zeros(inverse_size,
                                      device = gradient.device.type,
                                      dtype = filter_coeffs.dtype)
            extra = bool((inverse_size - start) % down)
            tmp = torch.zeros((inverse_size - start) // down + extra)
            tmp[:gradient.shape[0]] = gradient
            gradient_up[start :: down] = torch.clone(tmp)

        if (up == 1 and down == 1):
            out_x = gradient
        else:
            out_x = torch.conv1d(gradient_up.reshape(1, 1, inverse_size),
                               filter_coeffs.reshape(1, 1, filt_size))
            out_x = up * out_x.reshape(out_x.shape[-1])[::up]
        out_x = out_x[:x_size]

        if (up == 1 and down == 1):
            out_f = np.zeros(filt_coeffs.shape[0])
        else:
            out_f = torch.conv1d(gradient_up.reshape(1, 1, inverse_size),
                                 x_up.reshape(1, 1, x_up.shape[0]))
        out_f = out_f.reshape(out_f.shape[-1])[:filter_coeffs.shape[0]]

        return(out_x, out_f, None, None)


class Resample(Module):
    def __init__(self, up, down, filter_coeffs):
        super(Resample, self).__init__()
        self.up = up
        self.down = down
        self.filter_coeffs = filter_coeffs

    def forward(self, x):
        return FuncToyResample.apply(x, self.filter_coeffs, self.up, self.down)


def gradcheck_main(repetitions = 1):
    for i in range(repetitions):
        up = torch.randint(1, 20, (1,), requires_grad = False)
        down = torch.randint(1, 20, (1,), requires_grad = False)
        filter_size = np.random.randint(10,30)
        filter_coeffs = torch.randn(filter_size, requires_grad = True,
                                    dtype = torch.double,
                                    device = 'cpu')
        inputs = torch.randn(100, dtype = torch.double, requires_grad = True,
                             device = 'cpu')
        module = Resample(up, down, filter_coeffs)
        test = gradcheck(module, inputs, eps=1e-6, atol=1e-4)
        if not test:
            print(f'Are the gradients correct? {test}')
            print(f'Up: {up}, down: {down}')
    print(f'Are the gradients correct? {test}')


def scipy_check_main(repetitions = 1):
    for i in range(repetitions):
        x_size = np.random.randint(30, 100)
        filter_size = np.random.randint(5, 20)
        x = torch.randn(x_size)
        f = torch.randn(filter_size)
        up = torch.randint(1, 20, (1,))
        down = torch.randint(1, 20, (1,))
        module = Resample(up, down, f)
        scipy_resample = resample_poly(x, up, down, window = f.numpy())
        our_resample = module.forward(x)
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
    #scipy_check_main(10)
    gradcheck_main(10)
