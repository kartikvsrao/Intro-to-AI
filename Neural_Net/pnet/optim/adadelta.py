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
    "Adadelta", "adadelta"
]

class Adadelta(Optimizer):

    def __init__(self, params, lr, rho=0.95, eps=1e-8, name="Adadelta"):
        super().__init__(params, name)
        self.eps = eps
        self.lr = lr if isinstance(lr, Tensor) else constant(lr)
        self.rho = constant(rho, dtype=dtype.float32, name=self.name+"/rho")
        self.v = {
            v: parameter.zeros_like(v, name=self.name+"/m/"+v.name, requires_grad=False) for v in self.params
        }
    
    def _update(self, param):
        v = self.v[param]

        v.data *= self.rho.data
        v.data += (1-self.rho.data) * param.grad * param.grad
        
        param.data -= self.lr.data * param.grad / (np.sqrt(v.data) + self.eps)


adadelta = Adadelta
