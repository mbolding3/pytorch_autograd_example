#!/usr/bin/env python3


import torch
import numpy as np
import cupy as cp
from torch import flip
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
from math import gcd
from scipy.sparse import lil_matrix
#from scipy.signal import resample_poly
from cusignal import resample_poly


def get_start_index(length):
    if length <= 2:
        return 0
    return (length - 1) // 2


class FuncCusignalResample(Function):
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
            x_up = None
        else:
            x_up = torch.zeros(up * x_size, device = x.device.type,
                               dtype = x.dtype)
            # This is probably terrible for device data.
            x_up[::up] = up * torch.clone(x)
            window = filter_coeffs.numpy()
            if x.device == 'cuda':
                gpupath = True
                window = cp.array(window)
            else:
                gpupath = False
            x_out = resample_poly(x, up, down, window = window,
                                  gpupath = gpupath)
            out_len = torch.Tensor([len(x_out)])
            inverse_size = up * x_size + filt_size - 1
            inverse_size = torch.Tensor([inverse_size])
        ctx.save_for_backward(x_size_og, f_og, up_og, down_og, inverse_size,
                              out_len, x_up)
        return(torch.Tensor(cp.asnumpy(x_out)))

    @staticmethod
    def backward(ctx, gradient):
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
            tmp = torch.zeros(out_x_len)
            tmp[:gradient.shape[0]] = gradient
            gradient = tmp          
            gradient_up = torch.zeros(inverse_size,
                                      device = gradient.device.type,
                                      dtype = gradient.dtype)
            extra = bool((inverse_size - start) % down)
            tmp = torch.zeros((inverse_size - start) // down + extra)
            tmp[:gradient.shape[0]] = gradient
            gradient_up[start :: down] = torch.clone(tmp)

        # J_x up \times J_x conv
        if (up == 1 and down == 1):
            out_x = gradient
        else:
            out_x = torch.conv1d(gradient_up.reshape(1, 1, inverse_size),
                                 filter_coeffs.reshape(1, 1, filt_size))
            out_x = up * out_x.reshape(out_x.shape[-1])[::up]
        out_x = out_x[:x_size]

        # J_f conv
        if (up == 1 and down == 1):
            out_f = torch.zeros(filter_coeffs.shape[0],
                                device = gradient.device.type,
                                dtype = filter_coeffs.dtype)
        else:
            out_f = torch.conv1d(gradient_up.reshape(1, 1, inverse_size),
                                 x_up.type(gradient.dtype).\
                                 reshape(1, 1, x_up.shape[0]))
        out_f = out_f.reshape(out_f.shape[-1])[:filter_coeffs.shape[0]]

        return(out_x, out_f, None, None)


class Resample(Module):
    def __init__(self, up, down, filter_coeffs):
        super(Resample, self).__init__()
        self.up = up
        self.down = down
        self.filter_coeffs = filter_coeffs

    def forward(self, x):
        #return FuncToyResample.apply(x, self.filter_coeffs, self.up, self.down)
        return FuncCusignalResample.apply(x, self.filter_coeffs, self.up, self.down)


def accept_reps(f):
    def wrapper(repetitions = 1):
        for i in range(repetitions):
            f()
    return wrapper


@accept_reps
def gradcheck_main(eps=1e-3, atol=1e-3):
    '''
    Verifies that our backward method works.
    '''
    up = torch.randint(1, 20, (1,), requires_grad = False)
    down = torch.randint(1, 20, (1,), requires_grad = False)
    filter_size = np.random.randint(10,30)
    filter_coeffs = torch.randn(filter_size, requires_grad = True,
                                dtype = torch.double,
                                device = 'cpu')
    inputs = torch.randn(100, dtype = torch.double, requires_grad = True,
                         device = 'cpu')
    module = Resample(up, down, filter_coeffs)
    test = gradcheck(module, inputs, eps=eps, atol=atol)
    if not test:
        print(f'Are the gradients correct? {test}')
        print(f'Up: {up}, down: {down}')


@accept_reps
def forward_main():
    '''
    Verifies that our module agress with scipy's implementation
    on randomly generated examples.

    gpupath = True accepts numpy typed windows.
    gpupath = False accepts cupy types windows.
    '''
    gpupath = False
    x_size = np.random.randint(30, 100)
    filter_size = np.random.randint(5, 20)
    x = torch.randn(x_size)
    up = torch.randint(1, 20, (1,))
    down = torch.randint(1, 20, (1,))
    window = torch.randn(filter_size)
    module = Resample(up, down, window)
    scipy_resample = resample_poly(x, up, down, window = window.numpy(),
                                   gpupath = gpupath)
    our_resample = module.forward(x)
    if not np.allclose(scipy_resample, our_resample, atol=1e-4):
        print(f"up: {up}, down: {down}")
        print(f"scipy result: {scipy_resample[:10]}")
        print(f"our result: {our_resample[:10]}")
        return


if __name__ == '__main__':
    forward_main(100)
    gradcheck_main(100)
    print("tests complete")
