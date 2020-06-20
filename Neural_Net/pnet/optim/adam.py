# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

import numpy as np
from pnet import dtype
from pnet.optim.optimizer import Optimizer
from pnet.parameter import parameter, constant
from pnet.tensor import Tensor

__all__ = [
    "Adam", "adam"
]

class Adam(Optimizer):

    def __init__(self, params, lr, betas=(0.9, 0.999), eps=1e-8, name="Adam"):
        super().__init__(params, name)
        self.eps = eps
        self.lr = lr if isinstance(lr, Tensor) else constant(lr)
        self.t = parameter(0, dtype=dtype.int32, name=self.name+"/t", requires_grad=False)
        self.beta1 = constant(betas[0], dtype=dtype.float32, name=self.name+"/beta1")
        self.beta2 = constant(betas[1], dtype=dtype.float32, name=self.name+"/beta2")
        
        self.m = {
            v: parameter.zeros_like(v, name=self.name+"/m/"+v.name, requires_grad=False) for v in self.params
        }
        self.v = {
            v: parameter.zeros_like(v, name=self.name+"/v/"+v.name, requires_grad=False) for v in self.params
        }
    
    def step(self):
        self.t.data += 1
        scale = np.sqrt(1-self.beta2.data**self.t.data) / (1-self.beta1.data**self.t.data)
        self._lr = self.lr.data * scale
        super().step()

    def _update(self, param):
        m, v = self.m[param], self.v[param]

        if self.beta1.data == 0:
            m.data = param.grad
        else:
            m.data *= self.beta1.data
            m.data += (1-self.beta1.data) * param.grad

        if self.beta2.data == 0:
            v.data = param.grad * param.grad
        else:
            v.data *= self.beta2.data
            v.data += (1-self.beta2.data) * param.grad * param.grad
        
        param.data -= self._lr * m.data / (np.sqrt(v.data) + self.eps)


adam = Adam
