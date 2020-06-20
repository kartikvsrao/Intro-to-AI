from pnet.op import Op
import numpy as np

__all__ = [
    "abs"
]

class Abs(Op):
    def __init__(self, x, inplace):
        self.inplace = inplace
        super().__init__([x])

    def _forward(self):
        if self.inplace:
            self.mask = self.inputs[0].data < 0
            self.inputs[0].data[self.mask] *= -1
            return self.inputs[0].data
        return np.abs(self.inputs[0].data)
    
    def _backward(self, gradient):
        if self.inplace:
            gradient[self.mask] *= -1
        else:
            gradient = np.array(gradient)
            gradient[self.inputs[0].data < 0] *= -1
        return gradient

def abs(x, inplace=False):
    return Abs(x, inplace)
