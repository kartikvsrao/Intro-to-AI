from pnet.op import Op
import numpy as np

__all__ = [
    "add"
]

class Add(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2])
    
    def _forward(self):
        return np.add(self.inputs[0].data, self.inputs[1].data)
    
    def _backward(self, gradient):
        # Op.backward will solve the broadcasting matters
        if self.inputs[0].requires_grad:
            dx0 = np.multiply(gradient, np.ones_like(self.inputs[0].data))
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = np.multiply(gradient, np.ones_like(self.inputs[1].data))
        else:
            dx1 = None
        return [dx0, dx1]

def add(x1, x2):
    return Add(x1, x2)
