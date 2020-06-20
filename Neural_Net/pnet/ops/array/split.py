# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "split"
]

class Split(Op):
    def __init__(self, xs, indices_or_sections, axis):
        self.axis = axis
        xs.append(indices_or_sections)
        super().__init__(xs)

    def _forward(self):
        xs = [x.data for x in self.inputs[:-1]]
        return np.split(xs, self.inputs[-1].data, self.axis)
    
    def _backward(self, gradient):
        return np.concatenate(gradient, self.axis)

def split(xs, indices_or_sections, axis=0):
    return Split(xs, indices_or_sections, axis)
