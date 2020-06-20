# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
from pnet.core import enable_gradient, disable_gradient, requiring_gradient

__all__ = [
    "no_grad", "stop_grad"
]

class NoGrad(Op):

    def __enter__(self):
        if requiring_gradient():
            self._previous_requiring_gradient = True
            disable_gradient()
        else:
            self._previous_requiring_gradient = False

    def __exit__(self, type, value, traceback):
        if self._previous_requiring_gradient:
            enable_gradient()

    def __init__(self, x=None):
        if x is not None:
            super().__init__([x], requires_grad=False)
    
    def _forward(self):
        return self.inputs[0].data


no_grad = NoGrad
stop_grad = no_grad