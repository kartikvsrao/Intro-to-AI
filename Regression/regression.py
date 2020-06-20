#! /usr/bin/python3
# regression.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to Clemson University and the authors.
#
# Authors: Pei Xu (peix@g.clemson.edu) and Ioannis Karamouzas (ioannis@g.clemson.edu)

"""
In this assignment, you will implement linear and logistic regression
using the gradient descent method. To complete the assignment, please
modify the linear_regression(), and logistic_regression() functions.

The package `matplotlib` is needed for the program to run.
You should also try to use the 'numpy' library to vectorize
your code, enabling a much more efficient implementation of
linear and logistic regression. You are also free to use the
native 'math' library of Python.

All provided datasets are extracted from the scikit-learn machine learning library.
These are called `toy datasets`, because they are quite simple and small.
For more details about the datasets, please see https://scikit-learn.org/stable/datasets/index.html

Each dataset is randomly split into a training set and a testing set using a ratio of 8 : 2.
You will use the training set to learn a regression model. Once the training is done, the code
will automatically validate the fitted model on the testing set.
"""

# use math and/or numpy if needed
import math
import numpy as np
from copy import deepcopy

z = []
def build_z(d, feature_count):
    global z
    def makeCounter_rec(base):
        def incDigit(num, pos):
            new = num[:]
            if(num[pos] == base - 1):
                new[pos] = 0
                if(pos < len(num) - 1):
                    return incDigit(new, pos + 1)
            else:
                new[pos] = new[pos] + 1
            return new

        def inc(num):
            return incDigit(num, 0)
        return inc

    z = []
    terms = []
    base = int(feature_count)
    inc = makeCounter_rec(base)
    combo = [0] * int(d)
    z.append(combo)
    terms.append(combo)
    for i in range(base ** len(combo)):
        combo = inc(combo)
        z_ = sorted(deepcopy(combo))
        if z_ not in terms:
            terms.append(z_)
            z.append(z_)

"""
name   : zx_swap
desc   : Used for treating a degree n polynomial like a linear model.
         Groups together x's for each weight. For example, z0 corresponds to w0, and z[0] contains
         a list containing 0, which is the index for x0. All x's whose indicies are listed in z
         are multiplied together.
returns: z, otherwise all corresponding x's to a weight with a given index
"""
def zx_swap(index, X):
    global z
    total = float(1)
    for i in z[index]:
        total *= X[i]
    return total

"""
name: sq_err
desc: returns the squared error between a true type and predicted type.
"""
def sq_err(X,y,W,logistic=False):
    y_hat = h(W,X,logistic)
    return (y_hat - y) ** 2

"""
name   : J
desc   : applies a set of weights (W) against a data set.
returns: cost value of weights W
"""
def J(W, X_data, Y_data, logistic=False):
    total_sq_error = float(0)
    for i in range(len(X_data)):
        X = X_data[i]
        y = Y_data[i]
        total_sq_error += float(sq_err(X, y, W, logistic))
    return float(total_sq_error / (2 * len(X_data)))
"""
name   : h
desc   : the hypothesis function. The relation between inputs (X) and weights (W).
returns: y_hat (predicted y)
"""
def h(W, X, logistic=False):
    global z
    y_hat = 0.0
    for i in range(len(z)):
        y_hat += W[i] * zx_swap(i,X)
    if logistic:
        y_hat = (1 / (1 + math.exp((-1 * y_hat))))
    return y_hat

def linear_regression(x, y, logger=None):
    """
    Linear regression using full batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector.
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature.
    If you scale the cost function by 1/#samples, you should use as learning rate alpha=0.001, otherwise alpha=0.0001

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.

    Returns
    -------
    w: a 1D array
       linear regression parameters
    """

    """
    name   : apply_alpha
    desc   : Essentially the partial derivative of J. Applies alpha to a weight and the derivative to
             descend J.
    returns: New weight value for a weight w
    """
    def apply_alpha(index, X_data, Y_data, W, alpha):
        #init derivative
        derivative = float(0)
        #get weights
        w = W[index]
        #calculate partial derivative
        for i in range(len(X_data)):
            X = tuple(X_data[i])
            y = Y_data[i]
            diff = h(W,X,logistic=False) - y
            derivative += (diff * zx_swap(index, X))
        #return partial derivative of type float
        return w - (alpha * (derivative / float(len(X_data))))


    global z
    feature_count = len(x[0])
    #build feature_normalization
    build_z(1, feature_count)
    #initialize Weights array
    W = [0.0] * len(z)
    temp_W = deepcopy(W)
    #init alpha
    alpha = 0.3
    decay = 0.95

    #initialize cost function values for loop
    last_J = J(W, x, y, logistic=False) + 1
    current_J = J(W, x, y, logistic=False)
    J_change = current_J - last_J
    iterations = 0
    #loop for convergence
    while J_change < 0 and abs(J_change) > 1e-6:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i,x,y,W,alpha))
        #get weights
        W = deepcopy(temp_W)
        ### scale alpha by decay to slowly approach best weights
        alpha = float(decay * alpha)
        #calculate difference between the previous and current J values for convergence
        last_J = current_J
        current_J = J(W, x, y, logistic=False)
        J_change = current_J - last_J
        logger.log(iterations, current_J)

    return W


def logistic_regression(x, y, logger=None):
    """
    Logistic regression using batch gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector.
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature.
    In gradient descent, you should use as learning rate alpha=0.001

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.

    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """

    """
    name   : apply_alpha
    desc   : Essentially the partial derivative of J. Applies alpha to a weight and the derivative to
             descend J.
    returns: New weight value for a weight w
    """
    def apply_alpha(index, X_data, Y_data, W, alpha):
        #init derivative
        derivative = float(0)
        #get weights
        w = W[index]
        #calculate partial derivative
        for i in range(len(X_data)):
            X = tuple(X_data[i])
            y = Y_data[i]
            pred_typ = h(W,X,logistic=True)
            #sigmoid simulation
            if pred_typ >= 0.5:
                pred_typ = 1
            else: pred_typ = 0
            diff = pred_typ - y
            derivative += (diff * zx_swap(index, X))
        #return partial derivative of type float
        return w - (alpha * (derivative / float(len(X_data))))

    global z
    feature_count = len(x[0])
    #build feature_normalization
    build_z(1, feature_count)
    #initialize Weights array
    W = [0.0] * len(z)
    temp_W = deepcopy(W)
    #init alpha
    alpha = 0.001
    decay = 0.995

    #initialize cost function values for loop
    last_J = J(W, x, y, logistic=True) + 1
    current_J = J(W, x, y, logistic=True)
    J_change = current_J - last_J
    iterations = 0
    #loop for convergence
    while J_change < 0 and abs(J_change) > 1e-6:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i,x,y,W,alpha))
        #get weights
        W = deepcopy(temp_W)
        ### scale alpha by decay to slowly approach best weights
        alpha = float(decay * alpha) ### scale alpha by decay to slowly approach best weights
        #calculate difference between the previous and current J values for convergence
        last_J = current_J
        current_J = J(W, x, y, logistic=True)
        J_change = current_J - last_J
        logger.log(iterations, current_J)

    return W


def linear_regression_sgd(x, y, logger=None):
    """
    Linear regression using stochastic gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by x^T w, where x is a column vector.
    The intercept term can be ignored due to that x has been augmented by adding '1' as an extra feature.
    If you scale the cost function by 1/#samples, you should use as learning rate alpha=0.001, otherwise alpha=0.0001

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the target value for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.

    Returns
    -------
    w: a 1D array
       linear regression parameters
    """
    def apply_alpha(index, X_data, Y_data, W, alpha):
        #init derivative
        derivative = float(0)
        #get weights
        w = W[index]
        #calculate partial derivative
        for i in range(len(X_data)):
            X = tuple(X_data[i])
            y = Y_data[i]
            diff = h(W,X,logistic=False) - y
            derivative += (diff * zx_swap(index, X))
        #return partial derivative of type float
        return w - (alpha * (derivative / float(len(X_data))))




    global z
    feature_count = len(x[0])
    #build feature_normalization
    build_z(1, feature_count)
    #initialize Weights array
    W = [0.0] * len(z)
    temp_W = deepcopy(W)
    #init alpha
    alpha = 0.3
    decay = 0.9995

    #initialize cost function values for loop
    last_J = J(W, x, y, logistic=False) + 1
    current_J = J(W, x, y, logistic=False)
    J_change = current_J - last_J
    iterations = 0
    #loop for convergence
    while J_change < 0 and abs(J_change) > 1e-6:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i,x,y,W,alpha))
        #get weights
        W = deepcopy(temp_W)
        ### scale alpha by decay to slowly approach best weights
        alpha = float(decay * alpha)
        #calculate difference between the previous and current J values for convergence
        last_J = current_J
        current_J = J(W, x, y, logistic=False)
        J_change = current_J - last_J
        logger.log(iterations, current_J)

    return W

def logistic_regression_sgd(x, y, logger=None):
    """
    Logistic regression using stochastic gradient descent.
    A 1D array w should be returned by this function such that given a
    sample x, a prediction can be obtained by p = sigmoid(x^T w)
    with the decision boundary:
        p >= 0.5 => x in class 1
        p < 0.5  => x in class 0
    where x is a column vector.
    The intercept/bias term can be ignored due to that x has been augmented by adding '1' as an extra feature.
    In gradient descent, you should use as learning rate alpha=0.001

    Parameters
    ----------
    x: a 2D array of size [N, f+1]
       where N is the number of samples, f is the number of features
    y: a 1D array of size [N]
       It contains the ground truth label for each sample in x
    logger: a logger instance for plotting the loss
       Usage: logger.log(i, loss) where i is the number of iterations
       Log updates can be performed every several iterations to improve performance.

    Returns
    -------
    w: a 1D array
       logistic regression parameters
    """
    def apply_alpha(index, X_data, Y_data, W, alpha):
        #init derivative
        derivative = float(0)
        #get weights
        w = W[index]
        #calculate partial derivative
        for i in range(len(X_data)):
            X = tuple(X_data[i])
            y = Y_data[i]
            pred_typ = h(W,X,logistic=True)
            #sigmoid simulation
            if pred_typ >= 0.5:
                pred_typ = 1
            else: pred_typ = 0
            diff = pred_typ - y
            derivative += (diff * zx_swap(index, X))
        #return partial derivative of type float
        return w - (alpha * (derivative / float(len(X_data))))



    global z
    feature_count = len(x[0])
    #build feature_normalization
    build_z(1, feature_count)
    #initialize Weights array
    W = [0.0] * len(z)
    temp_W = deepcopy(W)
    #init alpha
    alpha = 0.001
    decay = 0.995

    #initialize cost function values for loop
    last_J = J(W, x, y, logistic=True) + 1
    current_J = J(W, x, y, logistic=True)
    J_change = current_J - last_J
    iterations = 0
    #loop for convergence
    while J_change < 0 and abs(J_change) > 1e-6:
        iterations += 1
        for i in range(len(W)):
            temp_W[i] = float(apply_alpha(i,x,y,W,alpha))
        #get weights
        W = deepcopy(temp_W)
        ### scale alpha by decay to slowly approach best weights
        alpha = float(decay * alpha) ### scale alpha by decay to slowly approach best weights
        #calculate difference between the previous and current J values for convergence
        last_J = current_J
        current_J = J(W, x, y, logistic=True)
        J_change = current_J - last_J
        logger.log(iterations, current_J)

    return W



if __name__ == "__main__":
    import os
    import tkinter as tk
    from app.regression import App

    import data.load
    dbs = {
        "Boston Housing": (
            lambda : data.load("boston_house_prices.csv"),
            App.TaskType.REGRESSION
        ),
        "Diabetes": (
            lambda : data.load("diabetes.csv", header=0),
            App.TaskType.REGRESSION
        ),
        "Handwritten Digits": (
            lambda : (data.load("digits.csv", header=0)[0][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))],
                      data.load("digits.csv", header=0)[1][np.where(np.equal(data.load("digits.csv", header=0)[1], 0) | np.equal(data.load("digits.csv", header=0)[1], 1))]),
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Breast Cancer": (
            lambda : data.load("breast_cancer.csv"),
            App.TaskType.BINARY_CLASSIFICATION
        )
     }

    algs = {
       "Linear Regression (Batch Gradient Descent)": (
            linear_regression,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Batch Gradient Descent)": (
            logistic_regression,
            App.TaskType.BINARY_CLASSIFICATION
        ),
        "Linear Regression (Stochastic Gradient Descent)": (
            linear_regression_sgd,
            App.TaskType.REGRESSION
        ),
        "Logistic Regression (Stochastic Gradient Descent)": (
            logistic_regression_sgd,
            App.TaskType.BINARY_CLASSIFICATION
        )
    }

    root = tk.Tk()
    App(dbs, algs, root)
    tk.mainloop()
