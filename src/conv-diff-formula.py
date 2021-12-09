#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd.gradcheck import gradcheck


class ConvNaive(Function):
    '''
    Function class not intended to contain learnable parameters.
    All we need to defined in this example is the differential
    with respect to the input.
    '''
    @staticmethod
    def forward(ctx, input, filter):
        # detach so we can cast to NumPy
        input = input.detach()
        filter = filter.detach()
        ctx.save_for_backward(input, filter)
        assert len(input.shape) == 1, 'Only 1d conv allowed.'
        assert len(filter.shape) == 1, 'Only 1d conv allowed.'
        input_np = input.numpy()
        filter_np = filter.numpy()
        if len(input_np) < len(filter_np):
            input_np, filter_np = filter_np, input_np
        n = len(input_np)
        k = len(filter_np)
        out = np.zeros(n-k+1)
        for i in range(n-k+1):
            out[i] = np.dot(input_np[i:i+k], filter_np)
        return torch.as_tensor(out, dtype=input.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, filter = ctx.saved_tensors
        filter_r = filter.numpy()[::-1]
        grad_np = grad_output.detach().numpy()
        n = len(input)
        k = len(filter)
        grad_input = np.zeros((n-k+1, n))
        for i in range(n-k+1):
            grad_input[i,n-k-i:n-i] = filter_r
        grad_input = (grad_input.transpose()@(grad_np)).transpose()
        return (torch.from_numpy(grad_input), None)


class ConvModule(Module):
    def __init__(self, filter_length):
        super(ConvModule, self).__init__()
        self.filter = Parameter(torch.randn(filter_length))

    def forward(self, input):
        return ConvNaive.apply(input, self.filter)


def main():
    '''
    In order for a tensor to be elligible for backward, you must set
    requires_grad = True apparently.
    '''
    filter_length = 3
    input_length = 10
    filter = torch.randn(filter_length)
    input = torch.randn(input_length, requires_grad = True)
    grad = torch.randn(input_length-filter_length+1)
    f = ConvNaive()
    result = f.apply(input, filter)
    print(f'Convolution output: {result}')
    result.backward(grad)
    print(f'Gradient (evaluated at random point): {input.grad}')


def gradcheck_main():
    filter_length = 3
    input_length = 10
    input = torch.randn(input_length, dtype=torch.double,
                        requires_grad = True)
    module = ConvModule(filter_length)
    test = gradcheck(module, input, eps=1e-6, atol=1e-4)
    print(f'Are the gradients correct: {test}')


if __name__ == '__main__':
    gradcheck_main()
