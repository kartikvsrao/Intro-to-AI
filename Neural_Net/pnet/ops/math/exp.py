from pnet.op import Op
import numpy as np

__all__ = [
    "exp"
]

class Exp(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.exp(self.inputs[0].data)
    
    def _backward(self, gradient):
        return np.multiply(gradient, self.data)

def exp(x):
    return Exp(x)
