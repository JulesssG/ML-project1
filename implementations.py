# -*- coding: utf-8 -*-

from helpers import *
import numpy as np

def least_squares_GD(y, tx, initial_w, iters, gamma, verbose=False):
    """
    Compute the linear regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    iters: The number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss
    """
    w = initial_w

    for n_iter in range(iters):
        gradient = compute_gradient_mse(y, tx, w)
        loss = compute_loss_mse(y, tx, w)

        if verbose:
            print_info(gradient, loss, n_iter)
        
        w = w - gamma * gradient

    loss = compute_loss_mse(y, tx, w)

    return w, loss


def least_squares_SGD(y, tx, initial_w, iters, gamma, verbose=False):
    """
    Compute the linear regression using stochastic gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    iters: The number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss
    """
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(iters):
        minibatch = np.random.choice(tx.shape[0], 1)

        minibatch_tx = tx[minibatch]
        minibatch_y = y[minibatch]

        gradient = compute_gradient_mse(minibatch_y, minibatch_tx, w)
        loss = compute_loss_mse(minibatch_y, minibatch_tx, w)

        if verbose:
            print_info(gradient, loss, n_iter)

        w = w - gamma * gradient

    loss = compute_loss_mse(y, tx, w)

    return w, loss


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
    mse = compute_loss_mse(y, tx, w)

    return w, mse


def logistic_regression(y, tx, initial_w, iters, gamma, verbose=False):
    """
    Compute the logistic regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    initial_w: The initial weights
    iters: The number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final loss by negative log likelihood
    """
    w = initial_w
    loss = compute_loss_logistic(y, tx, w)

    # start the logistic regression
    for n_iter in range(iters):
        gradient = compute_gradient_logistic(y, tx, w)

        w = w - gamma * gradient

        loss = compute_loss_logistic(y, tx, w)

        if verbose:
            print_info(gradient, loss, n_iter)
            
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, iters, gamma, verbose=False):
    """
    Compute the regularized logistic regression using gradient descent

    Parameters:
    y: The true values
    tx: The data
    lambda_: The regularizer parameter
    initial_w: The initial weights
    iters: The number of iterations
    gamma: Gamma (learning rate)

    Returns:
    w: The final weights
    loss: The final regularized loss by negative log likelihood
    """
    w = initial_w
    loss = compute_reg_loss_logistic(y, tx, w, lambda_)

    # start the logistic regression
    for n_iter in range(iters):
        gradient = compute_reg_gradient_logistic(y, tx, w, lambda_)

        w = w - gamma * gradient

        loss = compute_reg_loss_logistic(y, tx, w, lambda_)
        
        if verbose:
            print_info(gradient, loss, n_iter)

    return w, loss

