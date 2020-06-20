from pnet.op import Op
import numpy as np

__all__ = [
    "std"
]

class Std(Op):
    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__([x])

    def _forward(self):
        return np.std(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
    
    def _backward(self, gradient):
        mu = np.mean(self.inputs[0].data, axis=self.axis, keepdims=self.keepdims)
        dx = (1/self.inputs[0].shape[self.axis]) * (self.inputs[0].data - mu) / self.data
        if self.axis is not None and dx.ndim != gradient.ndim:
            gradient = np.expand_dims(gradient, axis=self.axis)
        return np.multiply(gradient, dx)

def std(x, axis=None, keepdims=False):
    return Std(x, axis, keepdims=False)
