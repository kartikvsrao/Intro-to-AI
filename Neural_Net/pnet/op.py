# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

import numpy as np

from pnet.core import executing_eagerly
from pnet.tensor import Tensor
from pnet.parameter import Constant

class Op(Tensor):

    def __init__(self, xs, requires_grad=None):
        self._xs = [x if isinstance(x, Tensor) else Constant(x) for x in xs]
        super().__init__(
            requires_grad=next((None for x in self._xs if x.requires_grad), False) \
                if requires_grad is None else requires_grad
        )
        if executing_eagerly():
            self.data = self._forward()

    def forward(self):
        if len(self.inputs) > 1:
            for x in self.inputs:
                x.forward()
        else:
            self.inputs[0].forward()
        return self.forward_()
    
    def forward_(self):
        self.data = self._forward()
        return self.data
        
    def backward(self, gradient=None):
        if self.requires_grad:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
                assert(gradient.shape == self.shape)
            elif gradient is None:
                gradient = np.ones_like(self.data)
            elif isinstance(gradient, np.ndarray):
                gradient = np.array(gradient)
                gradient.shape = self.shape
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient
            if len(self.inputs) > 1:
                dxs = self._backward(gradient)
                for x, dx in zip(self.inputs, dxs):
                    if x.requires_grad and dx is not None:
                        dx = self._verify_shape(x, dx)
                        x.backward(dx)
            else:
                dx = self._backward(gradient)
                if self.inputs[0].requires_grad and dx is not None:
                    dx = self._verify_shape(self.inputs[0], dx)
                    self.inputs[0].backward(dx)

    def backward_(self, gradient=None):
        if self.requires_grad:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
                assert(gradient.shape == self.shape)
            elif gradient is None:
                gradient = np.ones_like(self.data)
            elif isinstance(gradient, np.ndarray):
                gradient = np.array(gradient)
                gradient.shape = self.shape
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient
            if len(self.inputs) > 1:
                dxs = self._backward(gradient)
                for x, dx in zip(self.inputs, dxs):
                    if x.requires_grad and dx is not None:
                        dx = self._shape_verify(x, dx)
                        if x.grad is None:
                            x.grad = dx
                        else:
                            x.grad += dx
            else:
                dx = self._backward(gradient)
                if self.inputs[0].requires_grad and dx is not None:
                    dx = self._shape_verify(self.inputs[0], dx)
                    if self.inputs[0].grad is None:
                        self.inputs[0].grad = dx
                    else:
                        self.inputs[0].grad += dx
                    

    @property
    def inputs(self):
        return self._xs
        
    @staticmethod
    def _verify_shape(x, dx):
        if dx.shape != x.shape:
            if dx.size != x.size:   # broadcasting
                axis = 0
                for _ in range(dx.ndim):
                    axis -= 1
                    if axis + x.ndim < 0 or dx.shape[axis] != x.shape[axis]:
                        dx = np.sum(dx, axis=axis)
                        axis += 1
            dx.shape = x.shape
        return dx
