# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
from pnet.parameter import constant
import numpy as np

__all__ = [
    "cross_entropy"
]

class CrossEntropy(Op):
    def __init__(self, probs, labels, axis):
        self.axis = axis
        super().__init__([probs, labels])

    def _forward(self):
        assert(
            (self.inputs[1].ndim == 2 and (self.inputs[1].shape[1] == 1 or self.inputs[1].shape[1] == self.inputs[0].shape[self.axis])) \
            or self.inputs[1].ndim == 1
        )
        if self.inputs[1].ndim == 1:
            self.one_hot_labels = constant(np.eye(self.inputs[0].shape[self.axis])[self.inputs[1].data])
        elif self.inputs[1].shape[1] == 1:
            self.one_hot_labels = constant(np.eye(self.inputs[0].shape[self.axis])[np.reshape(self.inputs[1].data, [-1])])
        else:
            self.one_hot_labels = self.inputs[1]
        return -np.sum(np.log(self.inputs[0].data) * self.one_hot_labels.data, axis=self.axis)

    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            dx0 = -self.one_hot_labels.data / self.inputs[0].data
            dx0 = np.multiply(np.expand_dims(gradient, axis=self.axis), dx0)
        else:
            dx0 = None
        if self.inputs[1].requires_grad and self.inputs[1].ndim == 2:
            dx1 = np.multiply(np.expand_dims(gradient, axis=self.axis), -self.inputs[0].data)
        else:
            dx1 = None
        return [dx0, dx1]


def cross_entropy(probs, labels, axis=-1):
    return CrossEntropy(probs, labels, axis)
