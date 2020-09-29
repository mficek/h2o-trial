# sources:
# - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# - https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/

import numpy as np
import numba


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000000, fit_intercept=True, verbose=True):
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._fit_intercept = fit_intercept
        self._theta = None
        self._verbose = verbose

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self._theta = self._numba_fit(np.ascontiguousarray(X, dtype=np.float), y, self._learning_rate,
                                      self._num_iterations, self._fit_intercept, self._verbose)

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _numba_fit(X, y, learning_rate, num_iterations, fit_intercept, verbose):

        def add_intercept(X):
            intercept = np.ones((X.shape[0], 1))
            return np.concatenate((intercept, X), axis=1)

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def log_loss(y_hat, y):
            epsilon = 1e-5  # epsion to prevent division by zero if one of arguments is 1
            return -(1/y.size)*np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat + epsilon))  # in fact is a mean

        if fit_intercept:
            X = add_intercept(X)

        # weights initialization
        theta = np.zeros(X.shape[1])

        for i in range(num_iterations):
            z = np.dot(X, theta)
            y_hat = sigmoid(z)
            errors = y_hat - y
            gradient = np.dot(X.T, errors) / y.size
            theta -= learning_rate * gradient

            if (verbose and i % 10000 == 0):
                print('Iteration ', i, 'loss:', log_loss(y_hat, y), '\t')

        return theta


    def predict_proba(self, X):
        if self._fit_intercept:
            X = self._add_intercept(X)

        if self._theta is None:
            raise ValueError('Model should be fitted first.')
        return self._sigmoid(np.dot(X, self._theta))

    def predict(self, X, threshold=0.5):
        return int(self.predict_proba(X) >= threshold)


# test predictions
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]
coef = [-0.406605464, 0.852573316, -1.104746259]

lr = LogisticRegression()
X = np.array(dataset)[:, :-1]
y = np.array(dataset)[:, -1]
lr.fit(X, y)

lr.predict_proba(np.array([[2.7810836, 2.550537003]]))
lr.predict(np.array([[2.7810836, 2.550537003]]))