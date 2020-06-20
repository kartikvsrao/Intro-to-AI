from pnet.op import Op
import numpy as np

__all__ = [
    "maximum"
]

class Maximum(Op):
    def __init__(self, x1, x2):
        super().__init__([x1, x2])

    def _forward(self):
        return np.maximum(self.inputs[0].data, self.inputs[1].data)
    
    def _backward(self, gradient):
        # x0 >= x1 ? x0 : x1
        if self.inputs[0].requires_grad or self.inputs[1].requires_grad:
            mask = self.inputs[0].data >= self.inputs[1].data
        if self.inputs[0].requires_grad:
            dx0 = np.zeros_like(self.data)
            dx0[mask] = 1
            dx0 = np.multiply(gradient, dx0)
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = np.ones_like(self.data)
            dx1[mask] = 0
            dx1 = np.multiply(gradient, dx1)
        else:
            dx1 = None
        return [dx0, dx1]

def maximum(x1, x2):
    return Maximum(x1, x2)

