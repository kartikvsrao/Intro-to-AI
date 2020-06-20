# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "relu", "relu6"
]

class ReLU(Op):
    def __init__(self, x, max_value, negative_slope, threshold, inplace):
        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold
        self.inplace = inplace
        super().__init__([x])

    def _forward(self):
        x = self.inputs[0].data
        if not self.inplace:
            x = np.array(x)
        self.mask = x < self.threshold
        if self.max_value is not None and not np.isinf(self.max_value):
            max_mask = x >= self.max_value
            x[max_mask] = self.max_value
        else:
            max_mask = None
        if self.negative_slope == 0:
            x[self.mask] = 0
        else:
            x[self.mask] *= self.negative_slope
        if max_mask is not None:
            self.mask &= max_mask
        return x
    
    def _backward(self, gradient):
        if not self.inplace:
            gradient = np.array(gradient)
        gradient[self.mask] = 0
        return gradient


def relu(x, max_value=None, negative_slope=0.0, threshold=0.0, inplace=False):
    return ReLU(x, max_value, negative_slope, threshold, inplace)

def relu6(x, inplace=False):
    return ReLU(x, max_value=6, negative_slope=0.0, threshold=0.0, inplace=inplace)
