# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "mse_loss"
]

class MSELoss(Op):
    def __init__(self, preds, values):
        super().__init__([preds, values])

    def _forward(self):
        return np.mean(np.square(self.inputs[0].data - self.inputs[1].data)) * 0.5
    
    def _backward(self, gradient):
        if self.inputs[0].requires_grad or self.inputs[1].requires_grad:
            e = self.inputs[0].data - self.inputs[1].data
        if self.inputs[0].requires_grad:
            dx0 = e / self.inputs[0].size
            dx0 = np.multiply(gradient, dx0)
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = -e / self.inputs[0].size
            dx1 = np.multiply(gradient, dx1)
        else:
            dx1 = None
        return [dx0, dx1]

def mse_loss(preds, values):
    return MSELoss(preds, values)

 
