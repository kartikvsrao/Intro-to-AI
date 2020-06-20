# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "elu"
]

class ELU(Op):
    def __init__(self, x, alpha, inplace):
        self.alpha = alpha
        # FIXME inplace is useless due to the usage of numpy
        self.inplace = inplace
        super().__init__([x])

    def _forward(self):
        self.mask = self.inputs[0].data < 0
        return np.where(self.mask, 
            self.inputs[0].data,
            self.alpha*(np.exp(self.inputs[0].data)-1)
        )
        
    def _backward(self, gradient):
        gradient = np.where(self.mask,
            gradient,
            self.alpha*np.exp(self.inputs[0].data)*gradient
        )
        return gradient


def elu(x, alpha=0.2, inplace=False):
    return ELU(x, alpha, inplace)
