# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "l2_loss"
]

class L2Loss(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.sum(np.square(self.inputs[0].data)) * 0.5
    
    def _backward(self, gradient):
        dx = self.inputs[0].data
        return np.multiply(gradient, dx)

def l2_loss(x):
    return L2Loss(x)
