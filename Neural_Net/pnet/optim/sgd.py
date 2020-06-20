# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.optim.optimizer import Optimizer
from pnet.tensor import Tensor
from pnet.parameter import constant
__all__ = [
    "SGD", "sgd"
]

class SGD(Optimizer):

    def __init__(self, params, lr):
        self.lr = lr if isinstance(lr, Tensor) else constant(lr)
        super().__init__(params)

    def _update(self, param):
        param.data -= self.lr.data * param.grad


sgd = SGD
