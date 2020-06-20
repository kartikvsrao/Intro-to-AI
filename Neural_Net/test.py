import numpy as np

import pnet as nn
import data
from utils.logger import Logger


n_features = 24*24
n_classes=10
hidden_layer_level_1_size=256
hidden_layer_level_2_size=256

w1=nn.parameter(np.asmatrix(np.random.rand(n_features,hidden_layer_level_1_size) - 0.50))
#Weight2
# "w2": nn.parameter(np.asmatrix(np.random.rand(hidden_layer_level_1_size, hidden_layer_level_2_size) - 0.5))
# #Weight3
# "w3": nn.parameter(np.asmatrix(np.random.rand(hidden_layer_level_2_size, n_classes) - 0.50))
# #Bias1
# "b1": nn.parameter.zeros(hidden_layer_level_1_size)
# #Bias2
# "b2": nn.parameter.zeros(hidden_layer_level_2_size)
# #Bias3
# "b3": nn.parameter.zeros(n_classes)

print(np.random.rand(n_features,hidden_layer_level_1_size) - 1)
