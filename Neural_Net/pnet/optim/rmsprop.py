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
    "RMSProp", "rmsprop"
]

class RMSProp(Optimizer):

    def __init__(self, params, lr, decay=0.9, momentum=0.0, eps=1e-10, name="RMSProp"):
        super().__init__(params, name)
        self.eps = eps
        self.decay = constant(decay, dtype=dtype.float32, name=self.name+"/decay")
        self.momentum = constant(momentum, dtype=dtype.float32, name=self.name+"/momentum")
        self.lr = lr if isinstance(lr, Tensor) else constant(lr)
        self.m = {
            v: parameter.zeros_like(v, name=self.name+"/v/"+v.name, requires_grad=False) for v in self.params
        }
        self.v = {
            v: parameter.zeros_like(v, name=self.name+"/v/"+v.name, requires_grad=False) for v in self.params
        }
        if self.momentum.data != 0:
            self.mom = {
                v: parameter.zeros_like(v, name=self.name+"/v/"+v.name, requires_grad=False) for v in self.params
            }
        else:
            self.mom = None
    
    def _update(self, param):
        m, v = self.m[param], self.v[param] 

        m.data *= self.decay.data
        m.data += (1-self.decay.data) * param.grad

        v.data *= self.decay.data
        v.data += (1-self.decay.data) * param.grad * param.grad

        d = self.lr.data * param.grad / np.sqrt(v.data-np.square(m.data) + self.eps)
        if self.momentum.data == 0:
            param.data -= d
        else:
            mom = self.mom[param]
            mom.data *= self.momentum
            mom.data += d
            param.data -= mom.data


rmsprop = RMSProp
