import numpy as np
import math

from sympy.stats.sampling.sample_numpy import numpy

from ..utils import get_n_classes, label_to_onehot, onehot_to_label

N_FEATURES = 13
DATASET_SIZE = 1600


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """


    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """

        #HYPERPARAMETERS
        self.lr = lr
        self.max_iters = max_iters
        self.n_features = 13
        self.n_classes = 3
        self.convergence_treshold = 0.1

        #PARAMETERS
        self.W = np.zeros((self.n_classes,self.n_features))

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        weights = np.zeros(N_FEATURES)
        delta = 1000

        convergenceTreshold = 0.5  # To be modified later
        # Faire un treshold sur le gradiant
        n_iteration = 0
        delta = 1000
        scores = np.zeros((3, DATASET_SIZE))
        while (n_iteration < self.lr and delta < convergenceTreshold):
            # Addiction level {low, medium, high}
            sum = np.zeros(3)
            scores = self.predict(training_data)
            loss = 0
            for i in range(0, DATASET_SIZE):
                for classIndex in range(0, 3):
                    loss += training_labels[i, classIndex] * math.log(scores[i, classIndex])
            gradiant = training_data @ (scores - training_data)
            new_weights = weight - self.lr * gradiant
            delta = weight - new_weights
            weight = new_weights


        return scores

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        class_sum = np.zeros(self.n_classes)
        scores = np.zeros((3, DATASET_SIZE))
        for classificationClass in range(0, 3):
            # Faire le produit matriciel
            scores[classificationClass] = math.exp(self.W[classificationClass] @ test_data[classificationClass])
            class_sum[classificationClass] += np.sum(scores[classificationClass])
        for classificationClass in range(0, 3):
            scores[classificationClass] = scores[classificationClass] / class_sum[classificationClass].T
        return scores
