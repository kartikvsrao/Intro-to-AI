# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "one_hot"
]

class OneHot(Op):

    def __init__(self, indices, depth):
        self.depth = depth
        super().__init__([indices], requires_grad=False)
    
    def _forward(self):
        return np.eye(self.depth)[self.inputs[0].data]

def one_hot(indices, depth):
    return OneHot(indices, depth)
