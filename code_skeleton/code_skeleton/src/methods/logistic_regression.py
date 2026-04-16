import numpy as np
import math

from more_itertools.more import argmax
from sympy.stats.sampling.sample_numpy import numpy

from ..utils import get_n_classes, label_to_onehot, onehot_to_label, normalize_fn


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
        self.n_classes = 3
        self.convergence_treshold = 0.1
        self.n_features = 0
        #PARAMETERS
        self.W = np.ones((self.n_classes,self.n_features))

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.
        N SIZE DU DATASET
        D NOMBRE DE CLASS
        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """

        print("FIT CALLED RESET PARAMETERS")

        #Logique complète
        # predict renvoie les labels bruts et uniquement les labels
        #après on applique softmax et on renvoie la proabilité comme ça

        self.dataSetSize = training_data.shape[0]
        self.n_features = training_data.shape[1]
        self.W = np.ones((self.n_classes,self.n_features))
        #encoder ce merdier en one hot
        one_hot = np.eye(self.n_classes)[training_labels]

        prediction = np.zeros(training_data.shape[0])
        convergenceTreshold = 0.5  # To be modified later
        n_iteration = 0
        delta = 1000
        while (n_iteration < self.max_iters):
            prediction = self.predict(training_data)
            probabilityScores = self.computeSoftmax(training_data)
            loss = 0
            for i in range(0, self.dataSetSize):
                    loss -= np.sum(training_labels[i] * math.log(probabilityScores[i,prediction[i]]))
                    print("LOSS AT STEP",loss)
            gradiant = (training_data.T @ (probabilityScores - one_hot)).T
            new_weights = self.W - self.lr * gradiant
            delta = np.linalg.norm(gradiant)
            print(f"WEIGHT EVOLUTION AT STEP {n_iteration} {np.linalg.norm(self.W - new_weights)}")
            print(f"GRADIANT  {gradiant}")
            self.W = new_weights
            n_iteration +=1
        return prediction

    def predict(self, test_data):

        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        prediction = np.zeros((self.n_classes,test_data.shape[0]))

        for i in range (0,self.n_classes):
            prediction[i] = (self.W[i].T @ test_data.T).T
        prediction = prediction.T
        prediction = np.argmax(prediction,axis=1)
        return prediction.T


    def computeSoftmax(self,test_data):
        """
        Returns the predicted score with our current weights

        Arguments:
            test_data (np.array): test data of shape (N,D)

        Returns:
            pred_scores (np.array) : labels of shape(N,3)
        """

        prediction = test_data @ self.W.T

        predScores = np.exp(prediction)

        completeSum = np.sum(predScores, axis=1)
        predScores = predScores.T / completeSum.T

        return predScores.T
