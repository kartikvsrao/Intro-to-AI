# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "binary_cross_entropy"
]

class BinaryCrossEntropy(Op):
    def __init__(self, probs, labels, axis):
        self.axis = axis
        super().__init__([probs, labels])

    def _forward(self):
        assert(self.inputs[1].ndim == 1)
        self._log_p = np.log(self.inputs[0].data)
        self._log_1_p = np.log(1-self.inputs[0].data)
        return (self.inputs[1].data-1) * self._log_1_p \
            - self.inputs[1].data*self._log_p  

    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            dx0 = (1-self.inputs[1].data) / (1-self.inputs[0].data) \
                - self.inputs[1].data / self.inputs[0].data
            dx0 = np.multiply(gradient, dx0)
        else:
            dx0 = None
        if self.inputs[0].requires_grad:
            dx1 = np.multiply(gradient, self._log_1_p-self._log_p)
        else:
            dx1 = None
        return [dx0, dx1]


def binary_cross_entropy(probs, labels, axis=-1):
    return BinaryCrossEntropy(probs, labels, axis)
