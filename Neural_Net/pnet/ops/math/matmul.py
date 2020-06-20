from pnet.op import Op
import numpy as np

__all__ = [
    "matmul"
]

class Matmul(Op):

    def __init__(self, x1, x2):
        super().__init__([x1, x2])
    
    def _forward(self):
        # [N, K] x [K, M]
        return np.matmul(self.inputs[0].data, self.inputs[1].data)
    
    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            # [N, M] x [M, K]
            dx0 = np.matmul(gradient, np.transpose(self.inputs[1].data))
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            # [K, N] x [N, M]
            dx1 = np.matmul(np.transpose(self.inputs[0].data), gradient)
        else:
            dx1 = None
        return [dx0, dx1]

def matmul(x1, x2):
    return Matmul(x1, x2)
