import numpy as np
from mytorch.ops.function import Function, Context


class Add(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        return a.data + b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, grad_output


class Sub(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        return a.data - b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, -grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data * b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b.data, grad_output * a.data


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, a, b):
        ctx.save_for_backward(a, b)
        return a.data @ b.data

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, b = ctx.saved_tensors
        return grad_output @ b.data.T, a.data.T @ grad_output


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a, axis=None, keepdims=False):
        ctx.save_for_backward(a)
        return a.data.sum(axis=axis, keepdims=keepdims)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, = ctx.saved_tensors
        if a.grad is not None:
            a.grad += grad_output * np.ones_like(a.data)
        return grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a):
        ctx.save_for_backward(a)
        return np.maximum(0, a.data)

    @staticmethod
    def backward(ctx: Context, grad_output):
        a, = ctx.saved_tensors
        return grad_output * (a.data > 0)
