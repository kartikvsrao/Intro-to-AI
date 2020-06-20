from pnet.op import Op
import numpy as np

__all__ = [
    "greater"
]

class Greater(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2], requires_grad=False)
    
    def _forward(self):
        return np.greater(self.inputs[0].data, self.inputs[1].data)

def greater(x1, x2):
    return Greater(x1, x2)
