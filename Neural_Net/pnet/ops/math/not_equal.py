from pnet.op import Op
import numpy as np

__all__ = [
    "not_equal"
]

class NotEqual(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2], requires_grad=False)
    
    def _forward(self):
        return np.not_equal(self.inputs[0].data, self.inputs[1].data)

def not_equal(x1, x2):
    return NotEqual(x1, x2)
