from pnet.op import Op
import numpy as np

__all__ = [
    "add_n"
]

class AddN(Op):

    def __init__(self, xs):
        super().__init__(xs)
    
    def _forward(self):
        y = self.inputs[0].data
        for x in self.inputs[1:]:
            y += x.data
        return y
    
    def _backward(self, gradient):
        # Op.backward will solve the broadcasting matters
        dys = []
        for x in self.inputs:
            if x.requires_grad:
                dy = np.multiply(gradient, np.ones_like(x.data))
            else:
                dy = None
            dys.append(dy)
        return dys

def add_n(xs):
    return AddN(xs)
