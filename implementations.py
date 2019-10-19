# -*- coding: utf-8 -*-

import numpy as np

"""
FEATURE EXPANSION
"""
def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    N = x.shape[0]
    matrix = np.ones((N, 1))
    for i in range(1, degree+1):
        for feat in x.T:
            matrix = np.hstack((matrix, feat.reshape((N, 1))**i))
    
    return matrix

"""
LINEAR REGRESSION
"""
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


def gradient_descent(y, tx, initial_w, max_iters, gamma, verbose=False):
    """
    Compute the linear regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    max_iters: Max number of gradient descent iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss
    """
    w = initial_w

    for n_iter in range(max_iters):
        gradient = compute_gradient_mse(y, tx, w)
        
        w = w - gamma * gradient
        if n_iter % 100 == 0:
            print(compute_loss_mse(y, tx, w))

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma):
    """
    Compute the linear regression using stochastic gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    max_iters: Max number of gradient descent iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss
    """
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):
        minibatch = np.random.choice(tx.shape[0], 1)

        minibatch_tx = tx[minibatch]
        minibatch_y = y[minibatch]

        gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
        loss = compute_loss_mse(minibatch_y, minibatch_tx, w)

        w = w - gamma * gradient

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Compute the ridge regression using normal equations

    Parameters:
    y: The true values
    tx: The data
    lambda_: The regularizer parameter

    Returns:
    w: The computed weights
    mse: The mean square error loss
    """
    regularizer = 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1])

    A = tx.T @ tx + regularizer
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    mse = compute_mse(y, tx, w)

    return w, mse


def least_squares(y, tx):
    """
    Compute the least squares regression using normal equations

    Parameters:
    y: The true values
    tx: The data

    Returns:
    w: The computed weights
    mse: The mean square error loss
    """
    A = tx.T @ tx
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    mse = compute_mse(y, tx, w)

    return w, mse

"""
LOGISTIC REGRESSION
"""
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


def logistic_regression(y, tx, initial_w, max_iters, gamma, verbose=False):
    """
    Compute the logistic regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    max_iters: The max number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss by negative log likelihood
    """
    # init parameters
    threshold = 1e-8

    w = initial_w
    loss = compute_loss_logistic(y, tx, w)

    # start the logistic regression
    for iter in range(max_iters):
        gradient = compute_gradient_logistic(y, tx, w)

        w = w - gamma * gradient

        new_loss = compute_loss_logistic(y, tx, w)
        if np.abs(loss - new_loss) < threshold:
            loss = new_loss
            break

        loss = new_loss
        if verbose and iter % 10 == 0:
            print(f'Iteration : {iter} with loss {loss}')
    return w, loss


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


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, verbose=False):
    """
    Compute the regularized logistic regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    lambda_: The regularizer parameter
    initial_w: The initial weights
    max_iters: The max number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final regularized loss by negative log likelihood
    """
    # init parameters
    threshold = 1e-8

    w = initial_w
    loss = compute_reg_loss_logistic(y, tx, w, lambda_)

    # start the logistic regression
    for iter in range(max_iters):
        gradient = compute_reg_gradient_logistic(y, tx, w, lambda_)

        w = w - gamma * gradient

        new_loss = compute_reg_loss_logistic(y, tx, w, lambda_)

        if np.abs(loss - new_loss) < threshold:
            loss = new_loss
            break

        loss = new_loss
        
        if verbose and iter % 10 == 0:
            print(f'Iteration : {iter} with loss {loss}')
    return w, loss

