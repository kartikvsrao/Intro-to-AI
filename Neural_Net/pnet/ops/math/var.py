from pnet.op import Op
import numpy as np

__all__ = [
    "var"
]

class Var(Op):
    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__([x])

    def _forward(self):
        return np.var(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
    
    def _backward(self, gradient):
        mu = np.mean(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
        dx = (2/self.inputs[0].shape[self.axis]) * (self.inputs[0].data - mu)
        if self.axis is not None and dx.ndim != gradient.ndim:
            gradient = np.expand_dims(gradient, axis=self.axis)
        return np.multiply(gradient, dx)

def var(x, axis=None, keepdims=False):
    return Var(x, axis, keepdims=False)
