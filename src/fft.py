#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
import cupy as cp


def cp_to_torch(cp_output):
    return torch.as_tensor(cp.asnumpy(cp_output), device='cuda')


class FuncCusignalFFT(Function):
    '''
    Copying the implementation found here:
    https://github.com/locuslab/pytorch_fft/blob/b3ac2c6fba5acde03c242f50865c\
    964e3935e1aa/pytorch_fft/fft/autograd.py#L88

    1d only.
    '''

    @staticmethod
    def forward(ctx, input):
        input = cp.asarray(input.detach())
        cupy_fft = cp.fft.rfft(input)
        n = cupy_fft.size
        msg = 'Toy fft only defined for even inputs!'
        assert n % 2 == 0, msg
        out_r = cp.real(cupy_fft).reshape(n, 1)
        out_i = cp.imag(cupy_fft).reshape(n, 1)
        return cp_to_torch(out_r), cp_to_torch(out_i)

    @staticmethod
    def backward(ctx, grad_r, grad_i):
        grad_r = cp.asarray(grad_r.clone())
        grad_r[1:-1] /= 2
        grad_i = cp.asarray(grad_i.clone())
        grad_i[1:-1] /= 2
        n = grad_r.shape[0]
        grad = grad_r + 1j * grad_i
        grad = grad.reshape(n)
        out = cp.fft.irfft(grad)
        out = cp_to_torch(out)
        out *= 2 * (n - 1)
        return out


class CusignalFFT(Module):
    def __init__(self):
        super(CusignalFFT, self).__init__()

    def forward(self, input):
        return FuncCusignalFFT.apply(input)


def gradcheck_main():
    x = torch.randn(10, dtype = torch.double,
                    requires_grad = True, device = 'cuda')
    module = CusignalFFT()
    '''
    This is the best tolerance achievable. Setting them lower (eg. 1e-8)
    results in failure of the gradcheck.
    '''
    test = gradcheck(module, x, eps=1e-7, atol=1e-7)
    print(f'Are the gradients correct: {test}')


def forward_comparison_main():
    x = torch.randn(10, dtype = torch.double, device = 'cuda')
    module = CusignalFFT()
    this_r, this_i = module.forward(x)
    this_r = cp.array(this_r.reshape(6,1))
    this_i = cp.array(this_i.reshape(6,1))
    this_fft = cp.concatenate([this_r, this_i], axis=1)
    that_fft = torch.rfft(x, 1)
    test = np.allclose(this_fft, that_fft.cpu().numpy())
    print(f'Does this forward implementation agree with torch? {test}')


if __name__ == '__main__':
    gradcheck_main()
    forward_comparison_main()
