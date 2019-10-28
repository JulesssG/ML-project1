# -*- coding: utf-8 -*-

import numpy as np

def standardize(tx):
    """
    Standardize the data

    Parameters:
    tx: The data

    Returns:
    tx: The normalized data
    """
    mean = np.mean(tx, axis=0)
    std = np.std(tx, axis=0)
    tx = (tx - mean) / std

    return tx


def add_bias(tx):
    """
    Add bias to the data

    Parameters:
    tx: The data

    Returns:
    tx: The bias'd data
    """
    return np.hstack((np.ones((tx.shape[0], 1)), tx))


def build_poly(tx, degree):
    """
    Compute polynomial feature expansion of unbiased matrix tx

    Parameters:
    tx: The data without bias
    degree: The maximum degree of polynomial feature expansion

    Returns:
    The data with bias + polynomial feature expansion
    """
    N = tx.shape[0]
    matrix = np.ones((N, 1))

    for i in range(1, degree + 1):
        for feature in tx.T:
            matrix = np.hstack((matrix, feature.reshape((N, 1)) ** i))
    
    return matrix


def build_poly_cross_2(tx):
    """
    Compute polynomial feature expansion with the products across features
    up to degree 2 of unbiased matrix tx

    Parameters:
    tx: The data without bias

    Returns:
    The data with bias + polynomial feature expansion
    """
    N, F = tx.shape
    matrix = np.ones((N, 1))

    # Adding feature with degree 1
    matrix = np.hstack((matrix, tx))
    
    # Adding feature with degree 2
    cross_ind = []
    for i in range(F):
        for j in range(F):
            if j >= i:
                cross_ind.append([i, j])
    for i, j in cross_ind:
        matrix = np.hstack((matrix, (tx[:, i]*tx[:, j]).reshape(N, 1)))
        
    return matrix


def build_poly_cross_3(tx):
    """
    Compute polynomial feature expansion with the products across features
    up to degree 3 of unbiased matrix tx

    Parameters:
    tx: The data without bias

    Returns:
    The data with bias + polynomial feature expansion
    """
    N, F = tx.shape
    matrix = np.ones((N, 1))

    # Adding feature with degree 1
    matrix = np.hstack((matrix, tx))
    
    # Adding feature with degree 2
    cross_ind = []
    for i in range(F):
        for j in range(F):
            if j >= i:
                cross_ind.append([i, j])
    for i, j in cross_ind:
        matrix = np.hstack((matrix, (tx[:, i]*tx[:, j]).reshape(N, 1)))
        
    # Adding feature with degree 3
    cross_ind = []
    for i in range(F):
        for j in range(F):
            for k in range(F):
                if j >= i and k >= j:
                    cross_ind.append([i, j, k])
    for i, j, k in cross_ind:
        matrix = np.hstack((matrix, (tx[:, i]*tx[:, j]*tx[:, k]).reshape(N, 1)))
        
    return matrix


def feature_expansion(tx):
    """
    Expand the data by adding, sqrt, cos and sin. Assumes that the data is biased.

    Parameters:
    tx: The data

    Returns:
    Expanded data tx
    """
    x_copy = tx.copy()
    for feature in tx.T[1:]:
        x_copy = np.hstack((x_copy, (np.sqrt(np.abs(feature)) * np.sign(feature)).reshape(feature.shape[0], 1)))
        x_copy = np.hstack((x_copy, np.cos(feature).reshape((feature.shape[0], 1))))
        x_copy = np.hstack((x_copy, np.sin(feature).reshape((feature.shape[0], 1))))
    return x_copy


def compute_accuracy(y, tx, w):
    """
    Compute the accuracy of the logistic binary classification predictions. Predictions are in [0, 1]

    Parameters:
    y: The true values
    tx: The data
    w: The weights

    Returns:
    The accuracy
    """
    mapping = np.vectorize((lambda tx: -1 if 0 <= tx < 0.5 else 1))

    predictions = mapping(sigmoid(tx @ w))

    return np.sum(y == predictions)/(y.shape[0])


def compute_accuracy_linear_reg(y_true, y_pred):
    """
    Compute the accuracy of the linear binary classification predictions.

    Parameters:
    y_true: The true values
    y_pred: The predictions

    Returns:
    The accuracy
    """
    y_pred[y_pred >= 0] = 1
    y_pred[y_pred < 0] = -1
    return np.sum(y_true == y_pred)/(y_true.shape[0])


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
    t: The argument we want to apply sigmoid on, assume an array-like argument

    Returns:
    sigmoid(t)
    """
    t[t>50] = 50
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
    txw = tx @ w
    txw[txw>50] = 50
    s = np.sum(np.log(1 + np.exp(txw)))
    loss = s - y.T @ tx @ w

    return loss


def compute_gradient_logistic(y, tx, w, ):
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


def sanitize(x):
    """
    Sanitize the data matrix, i.e. normalize it without taking into account the -999 values and set the -999 values to 0

    Parameters:
    x: The data

    Returns
    sanitized_x: The sanitized data
    """
    sanitized_x = x.copy()
    na_values_x = sanitized_x == -999
    
    for i, feature in enumerate(sanitized_x.T):
        # The 23nd feature is a special feature we use later on to split the data in categories, so we don't touch it
        if i != 23:
            na_values = feature == -999
            known_values = ~na_values

            mean = np.mean(feature[known_values])
            std = np.std(feature[known_values])

            if len(feature[known_values] > 0):
                sanitized_x[:, i] = sanitized_x[:, i] - mean
                if std != 0:
                    sanitized_x[:, i] = sanitized_x[:, i] / std
    
    sanitized_x[na_values_x] = 0

    return sanitized_x


def split_data(x, y, ratio, seed=1):
    """
    Split the dataset based on the given ratio.

    Parameters:
    y: The true values
    x, The data
    ratio: The percentage the will be given to the training set

    Returns:
    x_train, x_test, y_train, y_test
    """
    np.random.seed(seed)
    N = x.shape[0]
    num_train = int(N*ratio)
    rand_indexes = np.random.choice(np.arange(N), size=N, replace=False)

    x_train = x[rand_indexes[:num_train]]
    y_train = y[rand_indexes[:num_train]]

    x_test = x[rand_indexes[num_train:]]
    y_test = y[rand_indexes[num_train:]]

    return x_train, x_test, y_train, y_test


def print_info(gradient, loss, n_iter):
    """
    Print verbose information for regressions

    Parameters:
    gradient: The gradient
    loss: The loss
    n_iter: The iteration the regression is at

    Returns:
    None
    """
    if n_iter % 250 == 0:
        norm_gradient = np.sqrt(np.sum(gradient**2))
        extract_loss = loss.reshape(-1)[0]
        print(f'Iteration : {n_iter} with loss {extract_loss} and gradient norm: {norm_gradient}')

