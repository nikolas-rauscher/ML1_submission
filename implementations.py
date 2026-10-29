"""The six CS-433 methods. Inputs use 1D y/w and a 2D design matrix.

MSE = mean((y-Xw)^2)/2. Logistic loss is mean cross entropy.
Regularization is lambda_ * ||w||^2, including any intercept column.
Returned losses exclude regularization, as required by the handout.
"""

import numpy as np


def _mse(y, tx, w):
    return np.mean((y - tx @ w) ** 2) / 2


def _logloss(y, tx, w):
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def _sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize MSE by full-batch GD; return final weights and MSE."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * (tx.T @ (tx @ w - y)) / len(y)
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Minimize MSE by SGD with batch size one; seed NumPy for repeatability."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        w -= gamma * tx[i] * (tx[i] @ w - y[i])
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Solve the normal equations; use a pseudoinverse if singular."""
    a, b = tx.T @ tx, tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_*||w||^2; return unpenalized MSE."""
    a = tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1])
    b = tx.T @ y
    try:
        w = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(a) @ b
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Binary logistic GD for labels in {0,1}; return mean cross entropy."""
    return reg_logistic_regression(y, tx, 0.0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Logistic GD with L2 penalty; return unpenalized mean cross entropy."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        gradient = tx.T @ (_sigmoid(tx @ w) - y) / len(y)
        w -= gamma * (gradient + 2 * lambda_ * w)
    return w, _logloss(y, tx, w)
