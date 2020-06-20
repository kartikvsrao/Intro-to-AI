# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "dropout"
]

class Dropout(Op):
    def __init__(self, x, p, inplace):
        self.inplace = inplace
        super().__init__([x, p])

    def _forward(self):
        if self.inputs[1].data == 0:
            return self.inputs[0].data
        if self.inplace:
            # drop mask
            self.mask = np.random.binomial(1, self.inputs[1].data, size=self.inputs[0].shape)
            self.inputs[0].data[self.mask] = 0
            return self.inputs[0].data / self.inputs[1].data
        # keep mask
        self.mask = np.random.binomial(1, 1-self.inputs[1].data, size=self.inputs[0].shape)
        self.mask /= self.inputs[1].data
        return np.multiply(self.inputs[0].data, self.mask)
    
    def _backward(self, gradient):
        if self.inputs[1].data == 0:
            return gradient
        if self.inplace:
            # drop mask
            gradient /= self.inputs[1].data
            gradient[self.mask] = 0
        else:
            # keep mask
            gradient = np.multiply(gradient, self.mask)
        return [gradient, None]

def dropout(x, p=0.5, inplace=False):
    return Dropout(x, p, inplace)
