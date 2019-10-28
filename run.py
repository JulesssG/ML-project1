# -*- coding: utf-8 -*-

from helpers import *
from implementations import *
from proj1_helpers import *
import numpy as np

# Load the data
y_train, x_train, _ = load_csv_data('data/train.csv')

# Sanitize the data and reshape the y vector
sanitized_x = sanitize(x_train)
sanitized_x = add_bias(sanitized_x)
y = y_train.reshape((y_train.shape[0], 1))

# Feature 23 (23 because we added bias) splits the data in different categories and we will treat them differently
feature_23 = sanitized_x[:, 23]
x_minus_23 = sanitized_x[:, np.array(range(sanitized_x.shape[1])) != 23]

# The data separated
x_sep = []
y_sep = []

# The categories
categories_23 = np.unique(feature_23)

for i in np.unique(feature_23):
    x_sep.append(x_minus_23[feature_23 == i, :])
    y_sep.append(y[feature_23 == i, :])

# Extend the data
extended_x_sep = [feature_expansion(e) for e in x_sep]

# The different weights for different categories
weights = []

for i, x_chunk in enumerate(extended_x_sep):
    # Ridge regression
    w_init = np.random.rand(x_chunk.shape[1], 1)
    w, loss = ridge_regression(y_sep[i], x_chunk, 0.0001)

    weights.append(w)

# Load the test data
_, x_test, ids_test = load_csv_data('data/test.csv')

# Sanitize the data
sanitized_x_t = sanitize(x_test)
sanitized_x_t = add_bias(sanitized_x_t)

# Split in different categories
feature_23_t = sanitized_x_t[:, 23]
x_minus_23_t = sanitized_x_t[:, np.array(range(sanitized_x_t.shape[1])) != 23]
x_sep_t = []
categories_23_t = np.unique(feature_23_t)

for i in np.unique(feature_23_t):
    x_sep_t.append(x_minus_23_t[feature_23_t == i, :])

# Extend data
extended_x_sep_t = [feature_expansion(e) for e in x_sep_t]

# Compute the predictions
y_sep_t = []

for i, w in enumerate(weights):
    y_t = predict_labels(w, extended_x_sep_t[i])
    y_sep_t.append(y_t)

N = x_test.shape[0]
predictions_t = np.zeros((N,1))

# Rebuild the predictions, in order, since we split them
for i, value in enumerate(categories_23_t):
    ind = np.arange(N)[feature_23_t == value]
    predictions_t[ind] = y_sep_t[i]

# Write the predictions to a csv file
OUTPUT_PATH = 'predictions-run.csv'
create_csv_submission(ids_test, predictions_t, OUTPUT_PATH)

