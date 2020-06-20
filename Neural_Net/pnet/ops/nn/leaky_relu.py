# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "leaky_relu", "lrelu"
]

class LeakyReLU(Op):
    def __init__(self, x, alpha, inplace):
        self.inplace = inplace
        self.alpha = alpha
        super().__init__([x])

    def _forward(self):
        self.mask = self.inputs[0].data < 0
        if self.inplace:
            self.inputs[0].data[self.mask] *= self.alpha
            return self.inputs[0].data
        return np.where(self.mask, 
            self.alpha*self.inputs[0].data,
            self.inputs[0].data
        )
        
    def _backward(self, gradient):
        if self.inplace:
            gradient[self.mask] *= self.alpha
        else:
            gradient = np.where(self.mask,
                self.alpha*gradient,
                gradient
            )
        return gradient


def leaky_relu(x, alpha=0.2, inplace=False):
    return LeakyReLU(x, alpha, inplace)

lrelu = leaky_relu
