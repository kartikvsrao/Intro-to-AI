from pnet.op import Op
import numpy as np

__all__ = [
    "pow"
]

class Pow(Op):
    def __init__(self, x, power):
        super().__init__([x, power])

    def _forward(self):
        return np.power(self.inputs[0].data, self.inputs[1].data)
    
    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            if self.inputs[1].data == 1:
                dx0 = np.ones_like(self.inputs[0].data)
            else:
                dx0 = np.multiply(self.inputs[1].data,
                    np.divide(self.data, self.inputs[0].data)
                )
            dx0 = np.multiply(gradient, dx0)
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = np.multiply(np.log(self.inputs[0].data), self.data)
            dx1 = np.multiply(gradient, dx1)
        else:
            dx1 = None
        return [dx0, dx1]

def pow(x, power):
    return Pow(x, power)
