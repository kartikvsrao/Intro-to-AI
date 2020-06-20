# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "tanh"
]

class Tanh(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return np.tanh(self.inputs[0].data)
    
    def _backward(self, gradient):
        dx = 1 - np.square(self.data)
        return np.multiply(gradient, dx)

def tanh(x):
    return Tanh(x)
