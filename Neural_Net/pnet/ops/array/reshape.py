# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "reshape"
]

class Reshape(Op):
    def __init__(self, x, shape, inplace):
        self.inplace = inplace
        self.output_shape = shape
        super().__init__([x])

    def _forward(self):
        self.input_shape = self.inputs[0].shape[:]
        if self.inplace:
            self.inputs[0].data.shape = self.output_shape
            return self.inputs[0].data
        return np.reshape(self.inputs[0].data, self.output_shape)
    
    def _backward(self, gradient):
        self.inputs[0].data.shape = self.input_shape
        if self.inplace:
            gradient.shape = self.input_shape
        else:
            gradient = np.reshape(gradient, self.input_shape)
        return gradient


def reshape(x, shape, inplace=False):
    return Reshape(x, shape, inplace)
