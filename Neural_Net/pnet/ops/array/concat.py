# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "concat", "concatenate", "cat"
]

class Concat(Op):
    def __init__(self, xs, axis):
        self.axis = axis
        super().__init__(xs)

    def _forward(self):
        self.input_shapes = [x.shape[self.axis] for x in self.inputs]
        xs = [x.data for x in self.inputs]
        return np.concatenate(xs, self.axis)
    
    def _backward(self, gradient):
        return np.split(gradient, self.input_shapes, axis=self.axis)

def concat(xs, axis):
    return Concat(xs, axis)

concatenate = concat
cat = concat
