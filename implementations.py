# -*- coding: utf-8 -*-

import numpy as np


def compute_loss(y, tx, w):
    return compute_mse(y, tx, w)

def compute_gradient(y, tx, w):
    return compute_gradient_mse(y, tx, w)


def compute_mse(y, tx, w):
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
    mse = (e @ e) / (2 * e.shape[0])

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


def gradient_descent(y, tx, initial_w, max_iters, gamma):
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
        gradient = compute_gradient(y, tx, w)
        
        w = w - gamma * gradient

    loss = compute_loss(y, tx, w)

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

        gradient = compute_gradient(minibatch_y, minibatch_tx, w)
        loss = compute_loss(minibatch_y, minibatch_tx, w)

        w = w - gamma * gradient

    loss = compute_loss(y, tx, w)

    return w, loss


def ridge_regression(y, tx, lambda_):
    """
    Compute the ridge regression using normal equations

    Parameters:
    y: The true values
    tx: The data
    lambda_: lambda

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
