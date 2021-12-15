#!/usr/bin/env python3


import torch
import sys
import math
import time
from class_autograd import LegendrePolynomial3
import numpy as np


class LegendreViaNumpy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        _input = np.array(input)
        _input = 0.5*(5*_input**3-3*_input)
        return(torch.Tensor(_input))

    @staticmethod
    def backward(ctx, grad):
        input, = ctx.saved_tensors
        return grad*1.5*(5*input**2-1)


def main():
    dtype = torch.float
    device = torch.device("cpu")
    x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
    y = torch.sin(x)
    a = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    b = torch.full((), -1.0, device=device, dtype=dtype, requires_grad=True)
    c = torch.full((), 0.0, device=device, dtype=dtype, requires_grad=True)
    d = torch.full((), 0.3, device=device, dtype=dtype, requires_grad=True)
    learning_rate = 5e-6
    N = 2000
    times = N*[0]
    for t in range(N):
        past = time.time()
        P3 = LegendrePolynomial3.apply
        y_pred = a + b * P3(c + d * x)
        loss = (y_pred - y).pow(2).sum()
        if t % 100 == 99:
            print(t, loss.item())
        loss.backward()
        with torch.no_grad():
            a -= learning_rate * a.grad
            b -= learning_rate * b.grad
            c -= learning_rate * c.grad
            d -= learning_rate * d.grad
            a.grad = None
            b.grad = None
            c.grad = None
            d.grad = None
        times[t] = time.time()-past

    print(f'Result: y = {a.item()} + {b.item()} * P3({c.item()} +'
          f' {d.item()} x)')
    print(f'Elapsed time: {np.average(times)}')


if __name__ == '__main__':
    main()
