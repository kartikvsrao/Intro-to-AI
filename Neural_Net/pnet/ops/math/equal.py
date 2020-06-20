from pnet.op import Op
import numpy as np

__all__ = [
    "equal"
]

class Equal(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2], requires_grad=False)
    
    def _forward(self):
        return np.equal(self.inputs[0].data, self.inputs[1].data)

def equal(x1, x2):
    return Equal(x1, x2)
