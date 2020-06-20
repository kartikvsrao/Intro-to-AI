from pnet.op import Op
import numpy as np

__all__ = [
    "log"
]

class Log(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.log(self.inputs[0].data)
    
    def _backward(self, gradient):
        return np.multiply(gradient, np.reciprocal(self.inputs[0].data))

def log(x):
    return Log(x)
