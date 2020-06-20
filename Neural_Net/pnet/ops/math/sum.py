from pnet.op import Op
import numpy as np

__all__ = [
    "sum"
]

class Sum(Op):
    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__([x])

    def _forward(self):
        return np.sum(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
    
    def _backward(self, gradient):
        dx = np.ones_like(self.inputs[0].data)
        if self.axis is not None and dx.ndim != gradient.ndim:
            gradient = np.expand_dims(gradient, axis=self.axis)
        return np.multiply(gradient, dx)

def sum(x, axis=None, keepdims=False):
    return Sum(x, axis, keepdims)
