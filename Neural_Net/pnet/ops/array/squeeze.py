# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "squeeze"
]

class Squeeze(Op):
    def __init__(self, x, axis, inplace):
        self.inplace = inplace
        self.axis = axis
        super().__init__([x])

    def _forward(self):
        if self.inplace:
            s = list(self.inputs[0].shape[:])
            del s[self.axis]
            self.inputs[0].data.shape = s
            return self.inputs[0].data
        return np.squeeze(self.inputs[0].data, self.axis)
    
    def _backward(self, gradient):
        if self.inplace:
            s = list(gradient.shape[:])
            if self.axis < 0:
                self.axis = gradient.ndim + 1 + self.axis
            s.insert(self.axis, 1)
            gradient.shape = s
        else:
            gradient = np.expand_dims(gradient, self.axis)
        return gradient

def squeeze(x, axis, inplace=False):
    return Squeeze(x, axis, inplace)
