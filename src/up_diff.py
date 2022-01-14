#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
from scipy.signal import correlate
from cusignal import correlate as cucorrelate
from cusignal import convolve as cuconvolve
import cupy as cp


def get_start_index(length):
    if length <= 2:
        return 0
    return (length - 1) // 2


class FuncUp(Function):
    @staticmethod
    def forward(ctx, x, up):
        x             = x.detach()
        up            = up.detach()
        ctx.save_for_backward(x, up)
        x_size = x.shape[0]
        up = int(up[0])
        x_up = torch.zeros(up * x_size, device = x.device.type,
                           dtype = x.dtype)
        x_up[::up] = up * x
        return(x_up)

    @staticmethod
    def backward(ctx, gradients):
        gradients = gradients.detach()
        _, up = ctx.saved_tensors
        out = up * torch.clone(gradients[::up])
        return(out, None)


class Up(Module):
    def __init__(self, up):
        self.up = up
        super(Up, self).__init__()

    def forward(self, x):
        return FuncUp.apply(x, self.up)


class FuncDown(Function):
    @staticmethod
    def forward(ctx, x, down, filt_size):
        x             = x.detach()
        down          = down.detach()
        filt_size     = filt_size.detach()
        ctx.save_for_backward(x, down, filt_size)
        down = int(down[0])
        x_size = x.shape[0]
        start = get_start_index(filt_size)
        x_out = torch.clone(x)[start::down]
        out_len = x_size // down + bool(x_size % down)
        x_out = x_out[:out_len]
        # Out length is out_len - start.
        return(x_out)

    @staticmethod
    def backward(ctx, gradients):
        x, down, filt_size = ctx.saved_tensors
        down = int(down[0])
        x_size = x.shape[0]
        start = get_start_index(filt_size)
        #out_len = x_size // down + bool(x_size % down)
        out = torch.zeros(x_size, dtype = x.dtype)
        out[start::down] = torch.clone(gradients)
        return(out, None, None)


class Down(Module):
    def __init__(self, down, filt_size):
        self.down = down
        self.filt_size = filt_size
        super(Down, self).__init__()

    def forward(self, x):
        return FuncDown.apply(x, self.down, self.filt_size)


def up_gradcheck_main():
    up = torch.randint(1,20, (1,), requires_grad = False)
    module = Up(up)
    inputs = torch.randn(10, dtype = torch.double, requires_grad = True)
    test = gradcheck(module, inputs)
    print(f'Are the gradients correct? {test}')


def down_gradcheck_main():
    down = torch.randint(1,20, (1,), requires_grad = False)
    filt_size = torch.randint(5, 20, (1,), requires_grad = False)
    module = Down(down, filt_size)
    inputs = torch.randn(10, dtype = torch.double, requires_grad = True)
    test = gradcheck(module, inputs)
    print(f'Are the gradients correct? {test}')


if __name__ == '__main__':
    down_gradcheck_main()
