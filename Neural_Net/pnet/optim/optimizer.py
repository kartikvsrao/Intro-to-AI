# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from abc import abstractmethod
import numpy as np

from pnet.parameter import Parameter

class Optimizer(object):
    
    def __init__(self, params, name=None):
        self.params = []
        for v in params:
            if v in self.params:
                continue
            if isinstance(v, Parameter):
                if v.requires_grad:
                    self.params.append(v)
                else:
                    print("[Warn] Ignore nontrainable parameter `" + v.name + "`.")
            else:
                raise ValueError("non-Parameter data is passed to Optimizer")
        # TODO name system
        self.name = name

    def zero_grad(self):
        for v in self.params:
            if v.grad is not None:
                v.grad.fill(0)

    def step(self):
        for v in self.params:
            if v.grad is not None:
                self._update(v)

    @abstractmethod
    def _update(self, v):
        pass
