"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC
import numpy as np

class SVM(object):
    """
    SVM method.
    """
    classifier = None
    def __init__(self, C, kernel, gamma=1., degree=1, coef0=0.):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            C (float): the weight of penalty term for misclassifications
            kernel (str): kernel in SVM method, can be 'linear', 'rbf' or 'poly' (:=polynomial)
            gamma (float): gamma prameter in rbf and polynomial SVM method
            degree (int): degree in polynomial SVM method
            coef0 (float): coef0 in polynomial SVM method
        """
        decesion_function_shape = 'ovo'
        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, decision_function_shape=decesion_function_shape)
        # print("[SVM] Initialized with: C={}, kernel={}, gamma={}, degree={}, coef0={}, decision_function_shape={}".format(C, kernel, gamma, degree, coef0, decesion_function_shape))

        
        
    def fit(self, training_data, training_labels):
        """
        Trains the model by SVM, then returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        self.classifier.fit(training_data, training_labels)
        return self.classifier.predict(training_data)
    
    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        pred_labels = self.classifier.predict(test_data)
        return pred_labels