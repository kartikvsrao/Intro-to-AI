from pnet.op import Op
import numpy as np

__all__ = [
    "greater_equal"
]

class GreaterEqual(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2], requires_grad=False)
    
    def _forward(self):
        return np.greater_equal(self.inputs[0].data, self.inputs[1].data)

def greater_equal(x1, x2):
    return GreaterEqual(x1, x2)
