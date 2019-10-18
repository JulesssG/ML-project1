# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    l = 2 * y.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = tx.T @ tx + l
    b = tx.T @ y
    
    w = np.linalg.solve(A, b)
    #w = np.linalg.solve(tx.T @ tx + (lambda_ * tx.shape[0]) * np.eye(tx.shape[1]), tx.T @ y)
    
    #mse = 1/(2*tx.shape[0]) * np.sum((y - tx @ w) ** 2)
    #rmse = np.sqrt(2 * mse)
    return w
