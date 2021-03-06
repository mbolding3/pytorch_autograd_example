#!/usr/bin/env python3


import torch
from torch.autograd import Function
import numpy as np
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.init import uniform_
from torch.autograd.gradcheck import gradcheck
from scipy.signal import correlate


class CorrNaive(Function):
    '''
    Function class not intended to contain learnable parameters.
    All we need to defined in this example is the differential
    with respect to the inputs.

    The implementation here is the same as that given in the
    Overleaf document. This is not at all computationally
    efficient but it is relatively easy to check that it
    matches the Overleaf formuals exactly.
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
        filter_np = filter.numpy()
        input_np = input.numpy()
        grad_np = grad_output.detach().numpy()
        n = len(input)
        k = len(filter)
        diff_input = np.zeros((n-k+1, n))
        diff_filter = np.zeros((n-k+1, k))
        for i in range(n-k+1):
            diff_input[i,i:i+k] = filter_np
            diff_filter[i,:] = input_np[i:i+k]
        grad_input = diff_input.transpose()@grad_np
        grad_filter = diff_filter.transpose()@grad_np
        return (torch.from_numpy(grad_input), torch.from_numpy(grad_filter))


class CorrModule(Module):
    '''
    This is all we have to do to add learnable parameters - 
    use the nn.Parameter function to declare them. There's probably
    a Pytorchy way to specify the weight intializer, but for now we
    do it uniformly randomly.
    '''
    def __init__(self, filter_features):
        super(CorrModule, self).__init__()
        self.filter_features = filter_features
        self.features = Parameter(torch.empty(filter_features))
        uniform_(self.features, -1, 1)

    def forward(self, input):
        return CorrNaive.apply(input, self.features)

    def extra_repr(self):
        return f'Filter: {self.features}'


def sanity_main():
    '''
    In order for a tensor to be elligible for backward, you must set
    requires_grad = True apparently.
    '''
    filter_length = 3
    input_length = 10
    filter = torch.randn(filter_length)
    input = torch.randn(input_length, requires_grad = True)
    grad = torch.randn(input_length-filter_length+1)
    f = CorrNaive()
    result = f.apply(input, filter)
    print(f'Convolution output: {result}')
    result.backward(grad)
    print(f'Gradient (evaluated at random point): {input.grad}')


def gradcheck_main():
    filter_length = 3
    input_length = 10
    input = torch.randn(input_length, dtype=torch.double,
                        requires_grad = True)
    module = CorrModule(filter_length)
    test = gradcheck(module, input, eps=1e-6, atol=1e-4)
    print(f'Are the gradients correct: {test}')


def scipy_main():
    filter_length = 3
    input_length = 10
    filter = torch.randn(filter_length)
    input = torch.randn(input_length)
    f = CorrNaive()
    result = f.apply(input, filter)
    scipy_result = correlate(input.numpy(), filter.numpy(), mode='valid')
    test = np.allclose(result, scipy_result)
    print(f'Does the function equal scipy.signal.correlate: {test}')


def backprop_main():
    '''
    Example of backprop. Just to prove that we can train
    our convolution layer using normal Pytorch ops. This code is
    copied from the polynomial fit example. It doesn't really make
    sense to use a correlation layer for this but who cares.
    '''
    N = 2000
    k = 10
    iters = 2000
    x = torch.linspace(-np.sqrt(10), np.sqrt(10), 2000)
    y = torch.sin(x)
    model = torch.nn.Sequential(
        CorrModule(10),
        torch.nn.Linear(1991,2000)
    )
    loss_fn = torch.nn.MSELoss(reduction='sum')
    learning_rate = 1e-6
    for t in range(iters):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learning_rate * param.grad
    corr_layer = model[0]
    print(f'Learned filter: {corr_layer.features.detach().numpy()}')


if __name__ == '__main__':
    scipy_main()
    gradcheck_main()
    backprop_main()
