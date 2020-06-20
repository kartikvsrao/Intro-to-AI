# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "transpose"
]

class Transpose(Op):
    def __init__(self, x, axes):
        self.axes = axes
        super().__init__(x)

    def _forward(self):
        return np.transpose(self.inputs[0].data, self.axes)
    
    def _backward(self, gradient):
        axes = [None] * gradient.ndim 
        for tar, src in enumerate(self.axes):
            axes[tar] = src
        return np.transpose(gradient, axes)

def transpose(x, axes):
    return Transpose(x, axes)
