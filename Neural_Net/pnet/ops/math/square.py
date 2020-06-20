from pnet.op import Op
import numpy as np

__all__ = [
    "square"
]

class Square(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.square(self.inputs[0].data)
    
    def _backward(self, gradient):
        dx = np.multiply(2, self.inputs[0].data)
        return np.multiply(gradient, dx)
        
def square(x):
    return Square(x)
