# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares solution."""
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    mse = 1/(2*tx.shape[0]) * np.sum((y - tx @ w) ** 2)
    return mse, w