# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "sigmoid"
]

class Sigmoid(Op):
    def __init__(self, x):
        super().__init__([x])

    def _forward(self):
        return 1 / (1+np.exp(-self.inputs[0].data))
    
    def _backward(self, gradient):
        dx = self.data * (1-self.data)
        return np.multiply(gradient, dx)

def sigmoid(x):
    return Sigmoid(x)
