# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

import numpy as np
from pnet.tensor import Tensor
import pnet.dtype as dtype

__all__ = [
    "parameter", "constant"
]

class Parameter(Tensor):

    @staticmethod
    def zeros(shape, dtype=dtype.float32, name=None, requires_grad=None):
        zeros = np.zeros(shape.data if isinstance(shape, Tensor) else shape, dtype=dtype)
        return parameter.from_numpy(zeros, name=name, requires_grad=requires_grad)

    @staticmethod
    def ones(shape, dtype=dtype.float32, name=None, requires_grad=None):
        ones = np.ones(shape.data if isinstance(shape, Tensor) else shape, dtype=dtype)
        return parameter.from_numpy(ones, name=name, requires_grad=requires_grad)

    @staticmethod
    def zeros_like(value, dtype=None, name=None, requires_grad=None):
        zeros = np.zeros_like(value.data if isinstance(value, Tensor) else value, dtype=dtype)
        return parameter.from_numpy(zeros, requires_grad=requires_grad)

    @staticmethod
    def ones_like(value, dtype=None, name=None, requires_grad=None):
        ones = np.ones_like(value.data if isinstance(value, Tensor) else value, dtype=dtype)
        return parameter.from_numpy(ones, requires_grad=requires_grad)

    @staticmethod
    def from_numpy(value, name=None, requires_grad=None):
        assert(isinstance(value, np.ndarray))
        p = Parameter(name=name, requires_grad=requires_grad)
        p.data = value
        return p

    def __init__(self, value=None, dtype=None, name=None, requires_grad=None):
        super().__init__(name=name, requires_grad=requires_grad)
        self.data = None if value is None else np.array(value, dtype=dtype)

    def assign_from_numpy(self, numpy_array):
        assert(isinstance(numpy_array, np.ndarray))
        self.data = numpy_array
        return self
    
    def assign_from_tensor(self, other_tensor):
        self.data = other_tensor.data
        return self

    def assign(self, value):
        if isinstance(value, Tensor):
            self.data = np.copy(value.data)
        else:
            self.data = np.array(value)
        return self

    def clone(self):
        tar = Parameter(self.data, name=self.name, requires_grad=False)
        if self.requires_grad:
            tar.requires_grad = self.requires_grad
            tar.grad = np.copy(self.grad)
        return tar
    
    def forward(self):
        return self.data

    def backward(self, gradient=None):
        if self.requires_grad:
            if isinstance(gradient, Tensor):
                gradient = gradient.data
                assert(gradient.shape == self.shape)
            elif gradient is None:
                # gradient = np.ones_like(self.data)
                gradient = np.zeros_like(self.data)
            elif isinstance(gradient, np.ndarray):
                gradient = np.array(gradient)
                gradient.shape = self.shape
            if self.grad is None:
                self.grad = gradient
            else:
                self.grad += gradient


class Constant(Parameter):

    def __init__(self, value, dtype=None, name=None):
        if isinstance(value, Tensor):
            value = value.data
        super().__init__(value, dtype=dtype, name=name, requires_grad=False)
    
    def assign(self, value):
        raise NotImplementedError
        
    @staticmethod
    def zeros(shape, dtype=dtype.float32, name=None):
        zeros = np.zeros(shape.data if isinstance(shape, Tensor) else shape, dtype=dtype)
        return parameter.from_numpy(zeros, name=name, requires_grad=False)

    @staticmethod
    def ones(shape, dtype=dtype.float32, name=None):
        ones = np.ones(shape.data if isinstance(shape, Tensor) else shape, dtype=dtype)
        return parameter.from_numpy(ones, name=name, requires_grad=False)

    @staticmethod
    def zeros_like(value, dtype=None, name=None):
        zeros = np.zeros_like(value.data if isinstance(value, Tensor) else value, dtype=dtype)
        return parameter.from_numpy(zeros, requires_grad=False)

    @staticmethod
    def ones_like(value, dtype=None, name=None):
        ones = np.ones_like(value.data if isinstance(value, Tensor) else value, dtype=dtype)
        return parameter.from_numpy(ones, requires_grad=False)

    @staticmethod
    def from_numpy(value, name=None):
        assert(isinstance(value, np.ndarray))
        p = Parameter(name=name, requires_grad=False)
        p.data = value
        return p

parameter = Parameter
constant = Constant
