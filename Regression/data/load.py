# load.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
# 
# Authors: Pei Xu (peix@g.clemson.edu)

import os
import numpy as np

def load(f, header=1):
    d = np.loadtxt(os.path.join(os.path.dirname(__file__), f), delimiter=',', skiprows=header)
    return d[:, :-1], d[:, -1]
