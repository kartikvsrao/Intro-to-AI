from pnet.op import Op
import numpy as np

__all__ = [
    "argmax"
]

class Argmax(Op):

    def __init__(self, x, axis):
        self.axis = axis
        super().__init__([x], requires_grad=False)
    
    def _forward(self):
        return np.argmax(self.inputs[0].data, axis=self.axis)

def argmax(x, axis=None):
    return Argmax(x, axis)
