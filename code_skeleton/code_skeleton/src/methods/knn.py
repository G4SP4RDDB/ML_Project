import numpy as np


class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=6, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        self.training_data = training_data
        self.training_labels = training_labels
        return self.training_labels
    

    def predict_label(self, neighbor_labels):
        """Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        if self.task_kind == "classification":
                return np.argmax(np.bincount(neighbor_labels.astype(int)))
        if self.task_kind == "regression":   
            return np.mean(neighbor_labels)
        
    def find_k_nearest_neighbors(self, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)
        return indices[:self.k]
    
    def kNN_one_example(self, unlabeled_example):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """
        # find distance of the single test example w.r.t. all training examples
        root = np.square(unlabeled_example - self.training_data)
        distances = np.sqrt(np.sum(root, axis = 1))
        
        # find the nearest neighbors
        nn_indices = self.find_k_nearest_neighbors(distances)
        
        # find the labels of the nearest neighbors
        neighbor_labels = self.training_labels[nn_indices]
        
        # Pick the most common
        best_label = self.predict_label(neighbor_labels)
        
        return best_label

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)
        """

        return np.apply_along_axis(func1d= self.kNN_one_example, axis=1, arr=test_data)
    