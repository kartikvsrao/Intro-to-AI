from pnet.op import Op
import numpy as np

__all__ = [
    "divide"
]

class Divide(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2])
    
    def _forward(self):
        return np.divide(self.inputs[0].data, self.inputs[1].data)
    
    def _backward(self, gradient):
        # Op.backward will solve the broadcasting matters
        if self.inputs[0].requires_grad:
            dx0 = np.multiply(gradient, np.reciprocal(self.inputs[1].data))
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = np.multiply(gradient, -self.data/self.inputs[1].data)
        else:
            dx1 = None
        return [dx0, dx1]

def divide(x1, x2):
    return Divide(x1, x2)
