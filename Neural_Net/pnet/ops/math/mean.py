from pnet.op import Op
import numpy as np

__all__ = [
    "mean"
]

class Mean(Op):
    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__([x])

    def _forward(self):
        return np.mean(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
    
    def _backward(self, gradient):
        if self.axis is None:
            dx = np.full_like(self.inputs[0].data, 1.0/self.inputs[0].size)
        else:
            dx = np.full_like(self.inputs[0].data, 1.0/self.inputs[0].shape[self.axis])
        if self.axis is not None and self.inputs[0].ndim != gradient.ndim:
            gradient = np.expand_dims(gradient, axis=self.axis)
        return np.multiply(gradient, dx)

def mean(x, axis=None, keepdims=False):
    return Mean(x, axis, keepdims)
