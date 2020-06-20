# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "prelu"
]

class PReLU(Op):
    def __init__(self, x, alpha):
        super().__init__([x, alpha])

    def _forward(self):
        self.mask = self.inputs[0].data < 0
        return np.where(self.mask, 
            self.inputs[1].data*self.inputs[0].data,
            self.inputs[0].data
        )
        
    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            dx = np.where(self.mask,
                self.inputs[1].data*gradient,
                gradient
            )
        else:
            dx = None
        if self.inputs[1].requires_grad:
            da = np.where(self.mask,
                self.inputs[0].data*gradient,
                np.zeros_like(self.inputs[1].data)
            )
        else:
            da = None
        return [dx, da]


def prelu(x, alpha):
    return PReLU(x, alpha)

