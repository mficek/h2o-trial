import logistic_regression
import numpy as np


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


def test_predictions():
    X = np.array(dataset)[:, :-1]
    y = np.array(dataset)[:, -1]

    lr = logistic_regression.LogisticRegression(learning_rate=0.01, num_iterations=10**6, verbose=10000)
    lr.coef_ = coef  # inject model coefficient
    assert all(lr.predict(X) == np.array(y, dtype=bool))
