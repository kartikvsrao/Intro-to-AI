# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from pnet.ops.math.abs import abs
from pnet.ops.math.sum import sum

import numpy as np

__all__ = [
    "l1_loss"
]

def l1_loss(x):
    return sum(abs(x, inplace=False))
