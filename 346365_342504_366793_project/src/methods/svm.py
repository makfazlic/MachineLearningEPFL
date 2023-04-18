"""
You are allowed to use the `sklearn` package for SVM.

See the documentation at https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
"""
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
        decesion_function_shape = 'ovr'
        self.classifier = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, decision_function_shape=decesion_function_shape)
        print("[SVM] Initialized with: C={}, kernel={}, gamma={}, degree={}, coef0={}, decision_function_shape={}".format(C, kernel, gamma, degree, coef0, decesion_function_shape))


    def validate_linear(self, training_data, training_labels, test_data, test_labels):
        kernel = 'linear'
        C = np.logspace(-2, 2, 5)
        Gamma = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        
        accs = []
        macrof1s = []

        best_c = 0
        best_gamma = 0
        best_acc = 0
        
        for c in tqdm(C, desc='C', leave=False, position=0):
            for gamma in tqdm(Gamma, desc="Gamma", leave=False, position=1):
                self.classifier = SVC(C=c, kernel=kernel, gamma=gamma)
                self.classifier.fit(training_data, training_labels)
                pred_labels = self.classifier.predict(test_data)
                acc = accuracy_fn(pred_labels, test_labels)
                macrof1 = macrof1_fn(pred_labels, test_labels)
                accs.append(acc)
                macrof1s.append(macrof1)
                if acc > best_acc:
                    best_acc = acc
                    best_c = c
                    best_gamma = gamma


        return best_c, best_gamma



    def validate_rbf(self, training_data, training_labels, test_data, test_labels):
        kernel = 'rbf'
        C = np.logspace(-2, 2, 5)
        Gamma = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        
        accs = []
        macrof1s = []

        best_c = 0
        best_gamma = 0
        best_acc = 0
        
        for c in tqdm(C, desc='C', leave=False, position=0):
            for gamma in tqdm(Gamma, desc="Gamma", leave=False, position=1):
                self.classifier = SVC(C=c, kernel=kernel, gamma=gamma)
                self.classifier.fit(training_data, training_labels)
                pred_labels = self.classifier.predict(test_data)
                acc = accuracy_fn(pred_labels, test_labels)
                macrof1 = macrof1_fn(pred_labels, test_labels)
                accs.append(acc)
                macrof1s.append(macrof1)
                if acc > best_acc:
                    best_acc = acc
                    best_c = c
                    best_gamma = gamma

        return best_c, best_gamma

    def validate_poly(self, training_data, training_labels, test_data, test_labels):
        kernel = 'poly'
        C = np.logspace(-2, 2, 5)
        Gamma = [0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
        Degree = [2, 3, 4, 5, 6, 7]
        Coef0 = np.logspace(-2, 2, 7)
        
        accs = []
        macrof1s = []

        best_c = 0
        best_gamma = 0
        best_degree = 0
        best_coef0 = 0
        best_acc = 0
        
        for c in tqdm(C, desc='C', leave=False, position=0):
            for gamma in tqdm(Gamma, desc="Gamma", leave=False, position=1):
                for degree in tqdm(Degree, desc="Degree", leave=False, position=2):
                    for coef0 in tqdm(Coef0, desc="Coef0", leave=False, position=3):
                        self.classifier = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
                        self.classifier.fit(training_data, training_labels)
                        pred_labels = self.classifier.predict(test_data)
                        acc = accuracy_fn(pred_labels, test_labels)
                        macrof1 = macrof1_fn(pred_labels, test_labels)
                        accs.append(acc)
                        macrof1s.append(macrof1)
                        if acc > best_acc:
                            best_acc = acc
                            best_c = c
                            best_gamma = gamma
                            best_degree = degree
                            best_coef0 = coef0

        return best_c, best_gamma, best_degree, best_coef0
        
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


# Redefine metrics
def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.

def macrof1_fn(pred_labels, gt_labels):
    """Return the macro F1-score."""
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels==val)
        
        tp = sum(predpos*gtpos)
        fp = sum(predpos*~gtpos)
        fn = sum(~predpos*gtpos)
        if tp == 0:
            continue
        else:
            precision = tp/(tp+fp)
            recall = tp/(tp+fn)

        macrof1 += 2*(precision*recall)/(precision+recall)

    return macrof1/len(class_ids)