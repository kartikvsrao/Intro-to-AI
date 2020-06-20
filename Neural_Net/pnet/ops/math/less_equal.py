from pnet.op import Op
import numpy as np

__all__ = [
    "less_equal"
]

class LessEqual(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2], requires_grad=False)
    
    def _forward(self):
        return np.less_equal(self.inputs[0].data, self.inputs[1].data)

def less_equal(x1, x2):
    return LessEqual(x1, x2)
