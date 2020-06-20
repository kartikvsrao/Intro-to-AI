# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "stack"
]

class Stack(Op):
    def __init__(self, xs, axis):
        self.axis = axis
        super().__init__(xs)

    def _forward(self):
        xs = [self.inputs[i].data for i in range(len(self.inputs))]
        return np.stack(xs, self.axis)
    
    def _backward(self, gradient):
        xs = np.split(gradient, gradient.shape[self.axis], axis=self.axis)
        return [np.squeeze(x, axis=self.axis) for x in xs]

def stack(xs, axis):
    return Stack(xs, axis)
