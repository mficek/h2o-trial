# sources:
# - https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# - https://machinelearningmastery.com/implement-logistic-regression-stochastic-gradient-descent-scratch-python/

import numpy as np
import numba


class LogisticRegression:

    def __init__(self,
                 learning_rate: float = 0.01,
                 num_iterations: int = 1000000,
                 fit_intercept: bool = True,
                 verbose: int = 10000):
        """
        Initialize logistic regresion class.

        :param learning_rate: learning rate for GD alorithm.
        :param num_iterations: number of iterations before stop.
        :param fit_intercept: add intercept if not exists.
        :param verbose: Show debug message every `verbose` iterations. Negative values or 0 disables verbosity.
        """
        self._learning_rate = learning_rate
        self._num_iterations = num_iterations
        self._fit_intercept = fit_intercept
        self._verbose = verbose
        self.coef_ = None
        self.loss_ = None

    @staticmethod
    def _add_intercept(X):
        '''
        Add column of ones to data.

        :param X: features
        '''
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.coef_, self.loss_ = self._numba_fit(np.ascontiguousarray(X, dtype=np.float), y, self._learning_rate,
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
            epsilon = 1e-9  # epsion to prevent division by zero if one of arguments is 1
            return -(1/y.size)*np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat + epsilon))  # in fact is a mean

        loss = np.zeros(num_iterations)*np.nan

        if fit_intercept:
            X = add_intercept(X)

        # weights initialization
        coef = np.zeros(X.shape[1])

        for i in range(num_iterations):
            z = np.dot(X, coef)
            y_hat = sigmoid(z)
            errors = y_hat - y
            gradient = np.dot(X.T, errors) / y.size
            coef -= learning_rate * gradient

            if (verbose > 0 and i % verbose == 0):
                loss[i] = log_loss(y_hat, y)
                print('Iteration:', i, 'loss:', loss[i], '\t')

        return coef, loss[~np.isnan(loss)]


    def predict_proba(self, X):
        if self._fit_intercept:
            X = self._add_intercept(X)

        if self.coef_ is None:
            raise ValueError('Model should be fitted first.')
        return self._sigmoid(np.dot(X, self.coef_))

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

