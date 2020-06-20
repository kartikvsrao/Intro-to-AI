# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

from .core import *
from .dtype import *
from .parameter import *

# array operations
from .ops.array.concat import *
from .ops.array.expand_dims import *
from .ops.array.reshape import *
from .ops.array.stack import *
from .ops.array.squeeze import *
from .ops.array.transpose import *
# TODO im2col
# element-wise operations
from .ops.math.abs import *
from .ops.math.exp import *
from .ops.math.log import *
from .ops.math.negative import *
from .ops.math.square import *
from .ops.math.sqrt import *
# unary operations
from .ops.math.mean import *
from .ops.math.std import *
from .ops.math.sum import *
from .ops.math.var import *
# binary operations
from .ops.math.add import *
from .ops.math.divide import *
from .ops.math.matmul import *
from .ops.math.maximum import *
from .ops.math.minimum import *
from .ops.math.multiply import *
from .ops.math.subtract import *
from .ops.math.pow import *
# misc mathematic operations
from .ops.math.add_n import *
# nondifferentiable operations:
from .ops.misc.no_grad import *
from .ops.misc.one_hot import *
from .ops.math.argmax import *
from .ops.math.equal import *
from .ops.math.not_equal import *
from .ops.math.greater import *
from .ops.math.greater_equal import *
from .ops.math.less import *
from .ops.math.less_equal import *
# neural network operations
from .ops.nn.binary_cross_entropy import *
from .ops.nn.cross_entropy import *
from .ops.nn.cross_entropy_loss import *
from .ops.nn.dropout import *
from .ops.nn.elu import *
from .ops.nn.l1_loss import *
from .ops.nn.l2_loss import *
from .ops.nn.leaky_relu import *
from .ops.nn.mse_loss import *
from .ops.nn.prelu import *
from .ops.nn.relu import *
from .ops.nn.sigmoid import *
from .ops.nn.sigmoid_cross_entropy_with_logits import *
from .ops.nn.softmax import *
from .ops.nn.softmax_cross_entropy_with_logits import *
from .ops.nn.softmax_cross_entropy_loss import *
# TODO batch norm
# TODO convolution
# TODO pooling
# optimizer
from .optim.adadelta import *
from .optim.adagrad import *
from .optim.adam import *
from .optim.rmsprop import *
from .optim.sgd import *
