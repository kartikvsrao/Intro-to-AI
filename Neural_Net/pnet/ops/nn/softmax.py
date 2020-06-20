# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "softmax"
]

class Softmax(Op):
    def __init__(self, x, axis=-1):
        self.axis = axis
        super().__init__([x])

    def _forward(self):
        if self.axis < 0:
            self.axis += self.inputs[0].ndim
        exps = np.exp(self.inputs[0].data-np.max(self.inputs[0].data, axis=self.axis, keepdims=True))
        return exps / np.sum(exps, axis=self.axis, keepdims=True)
    
    def _backward(self, gradient):
        y = self.data

        # numpy does not support stride dot
        # we use transpose to move the target axis to the last dimension such that
        # computation performs always on the last dimension
        if self.axis+1 != y.ndim:
            axes = list(range(self.inputs[0].ndim))
            axes.remove(self.axis)
            axes.append(self.axis)
            y = np.transpose(y, axes)
            gradient = np.transpose(gradient, axes)
        shape = y.shape[:]
        if len(shape) > 2:
            y.shape = [-1, y.shape[-1]]
            gradient.shape = [-1, y.shape[-1]]

        dot = np.matmul(np.expand_dims(y, axis=2), np.expand_dims(y, axis=1))
        dx = np.matmul(dot, np.expand_dims(gradient, axis=-1))
        dx.shape = y.shape
        dx = np.multiply(y, gradient) - dx
        dx.shape = shape
        
        if self.axis+1 != len(shape):
            axes = list(range(self.inputs[0].ndim-1))
            axes.insert(self.axis, self.inputs[0].ndim-1)
            dx = np.transpose(dx, axes)

        return dx

def softmax(x, axis=-1):
    return Softmax(x, axis)
