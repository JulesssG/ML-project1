# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train set accuracy")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Validation set accuracy")
    plt.xlabel("lambda")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of training and validation set")
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.figure(figsize=(7,5))
    plt.savefig("latex-report/validation")