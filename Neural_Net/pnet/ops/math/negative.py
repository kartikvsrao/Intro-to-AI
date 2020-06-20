from pnet.op import Op
import numpy as np

__all__ = [
    "negative"
]

class Negative(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.negative(self.inputs[0].data)
    
    def _backward(self, gradient):
        return np.negative(gradient)

def negative(x):
    return Negative(x)
