from pnet.op import Op
import numpy as np

__all__ = [
    "sqrt"
]

class Sqrt(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.sqrt(self.inputs[0].data)
    
    def _backward(self, gradient):
        dx = np.divide(0.5, self.data)
        return np.multiply(gradient, dx)
        
def sqrt(x):
    return Sqrt(x)
