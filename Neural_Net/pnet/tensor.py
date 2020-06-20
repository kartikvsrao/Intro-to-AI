# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from abc import abstractmethod
import numpy as np
from pnet.core import requiring_gradient

class Tensor(object):

    def __init__(self, name=None, requires_grad=None):
        self._requires_grad = requiring_gradient() if requires_grad is None else requires_grad
        # TODO name system for tensors
        self.name = "" if name is None else name
        self.data = None
        if self.requires_grad:
            self.grad = None

    @property
    def requires_grad(self):
        return self._requires_grad

    def __add__(self, rhs):
        import pnet.ops.math.add
        return pnet.ops.math.add.add(self, rhs)

    def __radd__(self, lhs):
        import pnet.ops.math.add
        return pnet.ops.math.add.add(lhs, self)

    # def __iadd__(self, rhs):
    #     self.data += rhs.item() if isinstance(rhs, Tensor) else rhs
    #     return self

    def __sub__(self, rhs):
        import pnet.ops.math.subtract
        return pnet.ops.math.subtract.subtract(self, rhs)
        
    def __rsub__(self, lhs):
        import pnet.ops.math.subtract
        return pnet.ops.math.subtract.subtract(lhs, self)

    # def __isub__(self, rhs):
    #     self.data -= rhs.item() if isinstance(rhs, Tensor) else rhs
    #     return self

    def __mul__(self, rhs):
        import pnet.ops.math.multiply
        return pnet.ops.math.multiply.multiply(self, rhs)

    def __rmul__(self, lhs):
        import pnet.ops.math.multiply
        return pnet.ops.math.multiply.multiply(lhs, self)

    # def __imul__(self, rhs):
    #     self.data *= rhs.item() if isinstance(rhs, Tensor) else rhs
    #     return self
    
    def __truediv__(self, rhs):
        import pnet.ops.math.divide
        return pnet.ops.math.divide.divide(self, rhs)

    def __rtruediv__(self, lhs):
        import pnet.ops.math.divide
        return pnet.ops.math.divide.divide(lhs, self)

    # def __itruediv__(self, rhs):
    #     self.data /= rhs.item() if isinstance(rhs, Tensor) else rhs
    #     return self

    def __matmul__(self, rhs):
        import pnet.ops.math.matmul
        return pnet.ops.math.matmul.matmul(self, rhs)

    def __rmatmul__(self, lhs):
        import pnet.ops.math.matmul
        return pnet.ops.math.matmul.matmul(lhs, self)

    # def __imatmul__(self, rhs):
    #     self.data @= rhs.item() if isinstance(rhs, Tensor) else rhs
    #     return self
    
    def __pow__(self, rhs):
        import pnet.ops.math.pow
        return pnet.ops.math.pow.pow(self, rhs)
    
    def __rpow__(self, lhs):
        import pnet.ops.math.pow
        return pnet.ops.math.pow.pow(lhs, self)

    def __neg__(self):
        import pnet.ops.math.negative
        return pnet.ops.math.negative.negative(self)
    
    def sum(self, axis=None, keepdims=False):
        import pnet.ops.math.sum
        return pnet.ops.math.sum.sum(self, axis, keepdims)
    
    def mean(self, axis=None, keepdims=False):
        import pnet.ops.math.mean
        return pnet.ops.math.mean.mean(self, axis, keepdims)

    def var(self, axis=None, keepdims=False):
        import pnet.ops.math.var
        return pnet.ops.math.var.var(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        import pnet.ops.math.std
        return pnet.ops.math.std.std(self, axis, keepdims)

    def squeeze(self, axis):
        import pnet.ops.array.squeeze
        return pnet.ops.array.squeeze.squeeze(self, axis)

    def expand_dims(self, axis):
        import pnet.ops.array.expand_dims
        return pnet.ops.array.expand_dims.expand_dims(self, axis)

    def unsqueeze(self, axis):
        import pnet.ops.array.expand_dims
        return pnet.ops.array.expand_dims.expand_dims(self, axis)

    def reshape(self, shape):
        import pnet.ops.array.reshape
        return pnet.ops.array.reshape.reshape(self, shape)

    def view(self, *shape):
        import pnet.ops.array.reshape
        return pnet.ops.array.reshape.reshape(self, shape)

    def zero_(self):
        if self.data is None:
            raise ValueError("Undefined value shape")
        self.data = np.zeros_like(self.data)
        return self

    def item(self):
        if isinstance(self.data, np.ndarray):
            try:
                ref = self.data[:]
            except:
                ref = np.copy(self.data)
            ref.setflags(write=False)
            return ref
        else:
            return self.data

    def gradient(self):
        if isinstance(self.grad, np.ndarray):
            try:
                ref = self.grad[:]
            except:
                ref = np.copy(self.grad)
            ref.setflags(write=False)
            return ref
        else:
            return self.grad
    
    @property
    def shape(self):
        return self.item().shape

    @property
    def dtype(self):
        return self.item().dtype
    
    @property
    def ndim(self):
        return self.item().ndim

    @property
    def size(self):
        return self.item().size

    # @shape.setter
    # def shape(self, s):
    #     self.item().shape = s

    @property
    def inputs(self):
        return []

    def _forward(self):
        return None

    def _backward(self, gradient):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
        
    def backward(self, gradient):
        raise NotImplementedError

    def assign(self, val):
        if isinstance(val, Tensor):
            self.data = np.copy(val.data)
        else:
            self.data = np.array(val)
        return self
    
    def assign_add(self, val):
        if isinstance(val, Tensor):
            self.data += val.data
        else:
            self.data += val
        return self

    def assign_sub(self, val):
        if isinstance(val, Tensor):
            self.data -= val.data
        else:
            self.data -= val
        return self

    def assign_mul(self, val):
        if isinstance(val, Tensor):
            self.data *= val.data
        else:
            self.data *= val
        return self
    