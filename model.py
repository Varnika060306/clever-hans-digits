"""
Softmax regression (multi-class logistic regression) written from scratch in NumPy.

For an input x (64 pixels) the model computes one score per class:
    scores = x @ W + b          W has shape (64, 10), b has shape (10,)
then turns the scores into probabilities with softmax.

I picked a linear model on purpose: every weight W[pixel, class] says how much
that pixel pushes the model towards that class. So we can just look at the
weights as 8x8 pictures and see what the model is paying attention to.
"""

import numpy as np


def softmax(scores):
    # subtract the max in each row so np.exp doesn't overflow (doesn't change the result)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def one_hot(labels, n_classes=10):
    out = np.zeros((len(labels), n_classes))
    out[np.arange(len(labels)), labels] = 1.0
    return out


class SoftmaxRegression:
    def __init__(self, n_features=64, n_classes=10, lr=0.5, epochs=500, l2=1e-3):
        self.lr = lr            # learning rate (step size for gradient descent)
        self.epochs = epochs    # how many passes over the training data
        self.l2 = l2            # small penalty that keeps weights from blowing up
        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros(n_classes)
        self.loss_history = []

    def predict_proba(self, X):
        return softmax(X @ self.W + self.b)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X, y):
        probs = self.predict_proba(X)
        n = len(y)
        # cross-entropy: average of -log(probability given to the correct class)
        data_loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        reg_loss = 0.5 * self.l2 * np.sum(self.W ** 2)
        return data_loss + reg_loss

    def fit(self, X, y):
        n = len(y)
        Y = one_hot(y, self.W.shape[1])

        for epoch in range(self.epochs):
            probs = self.predict_proba(X)

            # The nice thing about softmax + cross-entropy:
            # the gradient w.r.t. the scores is just (predicted probs - true one-hot).
            grad_scores = (probs - Y) / n               # shape (n, 10)

            grad_W = X.T @ grad_scores + self.l2 * self.W   # shape (64, 10)
            grad_b = grad_scores.sum(axis=0)                # shape (10,)

            # gradient descent step
            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

            self.loss_history.append(self.loss(X, y))

        return self

    def accuracy(self, X, y):
        return (self.predict(X) == y).mean()
