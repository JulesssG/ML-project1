# -*- coding: utf-8 -*-

import numpy as np

def standardize(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mean) / std
    return x


def add_bias(x):
    return np.hstack((np.ones((x.shape[0], 1)), x))


def build_poly(x, degree):
    """
    Compute polynomial feature expansion of unbiased matrix x

    Parameters:
    x: The data without bias
    degree: The maximum degree of polynomial feature expansion

    Returns:
    The data with bias + polynomial feature expansion
    """
    N = x.shape[0]
    matrix = np.ones((N, 1))

    for i in range(1, degree + 1):
        for feature in x.T:
            matrix = np.hstack((matrix, feature.reshape((N, 1)) ** i))
    
    return matrix


def compute_accuracy(y, x, w):
    mapping = np.vectorize((lambda x: -1 if 0 <= x < 0.5 else 1))

    predictions = mapping(sigmoid(x @ w))

    return np.sum(y == predictions)/(y.shape[0])

def compute_loss_mse(y, tx, w):
    """
    Compute the mean square error of the estimations compared to true values

    Parameters:
    y: The true values
    tx: The data
    w: The weights

    Returns:
    mse: The mean square error
    """
    e = y - tx @ w
    mse = (e.T @ e) / (2 * e.shape[0])

    return mse


def compute_gradient_mse(y, tx, w):
    """
    Compute the mean square error's gradient

    Parameters:
    y: The true values
    tx: The data
    w: The weights

    Returns:
    gradient: The computed gradient
    """
    e = y - tx @ w
    gradient = -1/tx.shape[0] * tx.T @ e

    return gradient


def sigmoid(t):
    """
    Apply sigmoid fuction

    Parameters:
    t: The argument we want to apply sigmoid on

    Returns:
    sigmoid(t)
    """
    sigmoid_t = (np.exp(t)) / (1 + np.exp(t))

    return sigmoid_t


def compute_loss_logistic(y, tx, w):
    """
    Compute the loss by negative log likelihood

    Parameters:
    y: The true values
    tx: The data
    w: The weights

    Returns:
    loss: The loss by negative log likelihood
    """
    exp = np.exp(tx @ w)
    log = np.log(1 + exp)
    s = np.sum(log)

    loss = s - y.T @ tx @ w

    return loss


def compute_gradient_logistic(y, tx, w):
    """
    Compute the gradient of the loss by negative log likelihood

    Parameters:
    y: The true values
    tx: The data
    w: The weights

    Returns:
    gradient: The gradient of the loss by negative log likelihood
    """
    gradient = tx.T @ (sigmoid(tx @ w) - y)

    return gradient


def compute_reg_loss_logistic(y, tx, w, lambda_):
    """
    Compute the regularized loss by negative log likelihood

    Parameters:
    y: The true values
    tx: The data
    w: The weights
    lambda_: The regularizer parameter

    Returns:
    loss: The regularized loss by negative log likelihood
    """
    exp = np.exp(tx @ w)
    log = np.log(1 + exp)
    s = np.sum(log)

    regularizer = lambda_ * np.linalg.norm(w) ** 2

    loss = s - y.T @ tx @ w + regularizer

    return loss


def compute_reg_gradient_logistic(y, tx, w, lambda_):
    """
    Compute the regularized gradient of the loss by negative log likelihood

    Parameters:
    y: The true values
    tx: The data
    w: The weights
    lambda_: The regularizer parameter

    Returns:
    gradient: The regularized gradient of the loss by negative log likelihood
    """
    regularizer = 2 * lambda_ * w

    gradient = tx.T @ (sigmoid(tx @ w) -y) + regularizer

    return gradient

