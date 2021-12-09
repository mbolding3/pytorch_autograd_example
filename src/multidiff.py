#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.autograd.gradcheck import gradcheck


class MultipleArgs(Function):
    '''
    u: a 2-vector
    v: a 4-vector
    How do we handle functions with more than one differentiable
    argument? This class seeks to answer that question.

    f(u, v) = [u_0 v_0 + u_1 v_1,
               u_0 v_2 + u_1 v_3].
    '''
    @staticmethod
    def forward(ctx, u, v):
        u_np = u.detach()
        v_np = v.detach()
        ctx.save_for_backward(u_np, v_np)
        u_np = u_np.numpy()
        v_np = v_np.numpy()
        out = np.zeros(2)
        out[0] = np.dot(u, v[:2])
        out[1] = np.dot(u, v[2:])
        return torch.as_tensor(out, dtype=u.dtype)

    @staticmethod
    def backward(ctx, gradients):
        '''
        The differentials here are nice and easy to compute.

        D_u : R^2 -> R^2,
        D_u = [[v_0, v_1],
               [v_2, v_3]]

        D_v : R^4 -> R^2,
        D_v = [[u_0, u_1,   0,   0],
               [  0,   0, u_0, u_1]]

        The question is how to interpret the gradients argument when
        one or both of the input tensors are marked as requiring
        grad.

        Remember that the output is supposed to be ( (D_u f)^T * gradients,
                                                     (D_v f)^T * gradients ).
        '''
        u, v = ctx.saved_tensors
        dtype = u.dtype
        u = u.numpy()
        v = v.numpy()
        gradients = gradients.numpy()
        du = [[v[0], v[1]],
              [v[2], v[3]]]
        dv = [[u[0], u[1], 0, 0],
              [0, 0, u[0], u[1]]]
        du = np.array(du)
        dv = np.array(dv)
        u_out = torch.as_tensor(du.transpose()@gradients, dtype=dtype)
        v_out = torch.as_tensor(dv.transpose()@gradients, dtype=dtype)
        return u_out, v_out


class MultiModule(Module):
    def __init__(self):
        super(MultiModule, self).__init__()

    def forward(self, u, v):
        return MultipleArgs.apply(u, v)


def basic_main():
    u = torch.randn(2, requires_grad=True)
    v = torch.randn(4, requires_grad=True)
    grad = torch.randn(2)
    f = MultipleArgs()
    result = f.apply(u, v)
    result.backward(grad)
    print(u.grad, v.grad)


def gradcheck_main():
    u = torch.randn(2, dtype=torch.double, requires_grad=True)
    v = torch.randn(4, dtype=torch.double, requires_grad=True)
    f = MultiModule()
    test = gradcheck(f, (u, v), eps=1e-10, atol=1e-6)
    print(f'Are the gradients correct: {test}')


if __name__ == '__main__':
    #basic_main()
    gradcheck_main()
