import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label


class LogisticRegression(object):
    """
    Logistic regression classifier with bias term.
    """

    def __init__(self, lr, max_iters=500, reg_lambda=0.01, eps=1e-8, gamma=1e-8):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            reg_lambda (float): regularization strength
            eps (float) : small constant to prevent division by zero, log(0) 
            gamma (float) : stopping the gradient descent
        """
        self.lr = lr
        self.max_iters = max_iters
        self.reg_lambda = reg_lambda
        self.eps = eps
        self.gamma = gamma

    def f_softmax(self, data, W, b):
        # Softmax function with bias term

        scores = np.dot(data, W) + b
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))

        probs = exp_scores / \
            (np.sum(exp_scores, axis=1, keepdims=True) + self.eps)
        return probs

    def loss_logistic_multi(self, data, labels, w, b):
        y = self.f_softmax(data, w, b)
        loss = -np.sum(labels * np.log(y + self.eps)) / data.shape[0]
        # reg_loss = 0.5 * self.reg_lambda * np.sum(w**2)
        return loss  # + reg_loss

    def gradient_logistic_multi(self, data, labels, W, b):
        return np.dot(np.transpose(data), (self.f_softmax(data, W, b) - labels)), np.sum(self.f_softmax(data, W, b) - labels, axis=0)

    def fit(self, training_data, training_labels):
        labels = label_to_onehot(training_labels)

        N, D = training_data.shape
        C = get_n_classes(training_labels)

        self.weights = np.random.normal(0, 0.1, (D, C))
        self.bias = np.zeros(C)

        prev_loss = np.inf
        for it in range(self.max_iters):
            gradient_w, gradient_b = self.gradient_logistic_multi(
                training_data, labels, self.weights, self.bias)
            self.weights = self.weights - self.lr * \
                (gradient_w + self.reg_lambda * self.weights)
            self.bias = self.bias - self.lr * gradient_b
            current_loss = self.loss_logistic_multi(
                training_data, labels, self.weights, self.bias)

            # Early stopping condition
            if np.abs(prev_loss - current_loss) < self.gamma:
                break
            prev_loss = current_loss
        predictions = self.predict(training_data)

        return self.predict(training_data)

    def predict(self, test_data):
        prob = self.f_softmax(test_data, self.weights, self.bias)
        pred_labels = np.zeros(test_data.shape[0])
        for i in range(pred_labels.shape[0]):
            pred_labels[i] = np.argmax(prob[i])
        return pred_labels
