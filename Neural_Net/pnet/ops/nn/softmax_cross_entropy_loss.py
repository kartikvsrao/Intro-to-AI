# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.op import Op
from pnet.ops.nn.softmax_cross_entropy_with_logits import SoftmaxCrossEntropyWithLogits
from pnet.ops.nn.cross_entropy import CrossEntropy
from pnet.parameter import constant
import numpy as np

__all__ = [
    "softmax_cross_entropy_loss"
]

def softmax_cross_entropy_loss(_stub=None, probs=None, logits=None, labels=None, axis=-1, reduction="mean"):
    reduction = reduction.lower()
    assert(reduction in ["mean", "sum"])
    assert((probs is None) != (logits is None))
    if probs is None:
        entropy = SoftmaxCrossEntropyWithLogits(logits, labels, axis)
        probs = entropy.probs
    else:
        entropy = CrossEntropy(probs, labels, axis)
    if reduction == "mean":
        loss = entropy.mean()
    else:
        loss = entropy.sum()
    return loss, probs

