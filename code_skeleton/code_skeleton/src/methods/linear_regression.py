import numpy as np
from numpy.linalg import pinv as pinv


class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self, w = None):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.w = w

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        N = training_labels.shape[0]
        X = np.hstack([np.ones((N, 1)), training_data]) #bias included in X

        self.w = pinv(X) @ training_labels #pseudo-inverse (X^T X)^-1 X^T @ y, the closed-form solution

        pred_labels = X @ self.w
        return pred_labels

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        N = test_data.shape[0]
        X_test = np.hstack([np.ones((N, 1)), test_data])

        pred_labels = X_test @ self.w

        return pred_labels
