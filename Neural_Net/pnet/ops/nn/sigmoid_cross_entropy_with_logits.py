# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
import numpy as np

__all__ = [
    "sigmoid_cross_entropy_with_logits",
    "binary_cross_entropy_with_logits"
]
class SigmoidCrossEntropyWithLogits(Op):
    def __init__(self, logits, labels):
        self._probs = None
        super().__init__([logits, labels])

    def _forward(self):
        #   - y log(1/(1+e^-x)) - (1-y) log(1-1/(1+e^-x))
        # = y log(1+e^-x) - (1-y) * (log(e^-x) - log(1+e^-x))
        # = y log(1+e^-x) + (1-y) * (x + log(1+e^-x))
        # = x - x*y + log(1+e^-x))
        # = log(e^x) - x*y + log(1+e^-x))
        # = - x*y + log(1+e^x)
        # = max(x, 0) - x*y + log(1+e^-|x|)
        # to prevent e^x overflow
        return np.maximum(self.inputs[0].data, 0) \
            - self.inputs[0].data*self.inputs[1].data \
            + np.log(1 + np.exp(-np.abs(self.inputs[0].data)))

    def _backward(self, gradient):
        if self.inputs[0].requires_grad:
            dx0 = self.probs - self.inputs[1].data
            dx0 = np.multiply(gradient, dx0)
        else:
            dx0 = None
        if self.inputs[1].requires_grad:
            dx1 = np.log(1-self.probs) - np.log(self.probs)
            dx1 = np.multiply(gradient, dx0)
        else:
            dx1 = None
        return [dx0, dx1]
    
    @property
    def probs(self):
        if self._probs is None:
            self._probs = 1 / (1+np.exp(-self.inputs[0].data))
        return self._probs

def sigmoid_cross_entropy_with_logits(logits, labels):
    return SigmoidCrossEntropyWithLogits(logits, labels)

binary_cross_entropy_with_logits = sigmoid_cross_entropy_with_logits
