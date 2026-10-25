"""The six machine learning methods required by Project 1 (CS-433).

Every method returns the tuple ``(w, loss)`` where ``w`` is the last weight
vector reached by the method and ``loss`` is the value of the corresponding
cost function evaluated at that ``w``.

Conventions used throughout this file:

* the mean squared error is ``L(w) = 1/(2N) * sum_n (y_n - x_n^T w)^2``;
* the logistic loss is the mean negative log-likelihood
  ``L(w) = 1/N * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]`` with
  labels ``y in {0, 1}``;
* the losses returned by the regularised methods (``ridge_regression`` and
  ``reg_logistic_regression``) do **not** include the penalty term;
* all weight vectors are 1D arrays of shape ``(D,)``;
* linear systems are solved with ``np.linalg.solve``, never by explicitly
  inverting a matrix.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Losses and gradients
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the mean squared error loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        The scalar MSE loss 1/(2N) * ||y - tx @ w||^2.
    """
    e = y - tx.dot(w)
    return e.dot(e) / (2 * y.shape[0])


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the mean squared error loss.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        numpy array of shape (D,), the gradient -1/N * tx.T @ (y - tx @ w).
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / y.shape[0]


def sigmoid(t):
    """Apply the logistic function element-wise, in a numerically stable way.

    Args:
        t: numpy array of any shape.

    Returns:
        numpy array of the same shape, with values in (0, 1).
    """
    # exp(-|t|) never overflows; the two branches are algebraically identical.
    positive = t >= 0
    out = np.empty_like(t, dtype=float)
    exp_neg = np.exp(-np.abs(t))
    out[positive] = 1.0 / (1.0 + exp_neg[positive])
    out[~positive] = exp_neg[~positive] / (1.0 + exp_neg[~positive])
    return out


def compute_logistic_loss(y, tx, w):
    """Compute the average negative log-likelihood of the logistic model.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        The scalar loss 1/N * sum_n [log(1 + exp(x_n^T w)) - y_n * x_n^T w].
    """
    z = tx.dot(w)
    # np.logaddexp(0, z) computes log(1 + exp(z)) without overflowing.
    return np.mean(np.logaddexp(0.0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the average logistic loss.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        numpy array of shape (D,), the gradient 1/N * tx.T @ (sigmoid(tx @ w) - y).
    """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) / y.shape[0]


# ---------------------------------------------------------------------------
# The six required methods
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using full-batch gradient descent on the MSE.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of gradient descent steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after max_iters steps.
        loss: the MSE evaluated at w.
    """
    w = np.asarray(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_mse_gradient(y, tx, w)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent on the MSE.

    At each iteration a single data point is drawn uniformly at random
    (mini-batch size of 1, as required by the project statement) and one
    gradient step is taken along its gradient.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of SGD steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after max_iters steps.
        loss: the MSE over the whole data set, evaluated at w.
    """
    w = np.asarray(initial_w, dtype=float)
    num_samples = y.shape[0]
    for _ in range(max_iters):
        n = np.random.randint(num_samples)
        # Slicing keeps the batch dimension, so the gradient formula is unchanged.
        w = w - gamma * compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression solved with the normal equations.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.

    Returns:
        w: numpy array of shape (D,), the least squares solution.
        loss: the MSE evaluated at w.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression solved with the normal equations.

    Minimises ``1/(2N) * ||y - tx w||^2 + lambda_ * ||w||^2``, whose normal
    equations are ``(tx^T tx + 2 N lambda_ I) w = tx^T y``.

    Args:
        y: numpy array of shape (N,), the labels.
        tx: numpy array of shape (N, D), the feature matrix.
        lambda_: float, the regularisation parameter.

    Returns:
        w: numpy array of shape (D,), the ridge solution.
        loss: the MSE evaluated at w, without the penalty term.
    """
    num_samples, num_features = tx.shape
    a = tx.T.dot(tx) + 2.0 * num_samples * lambda_ * np.eye(num_features)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using full-batch gradient descent.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of gradient descent steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after max_iters steps.
        loss: the average negative log-likelihood evaluated at w.
    """
    w = np.asarray(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    return w, compute_logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """L2-regularised logistic regression using full-batch gradient descent.

    Minimises ``L(w) + lambda_ * ||w||^2`` where ``L`` is the average negative
    log-likelihood, so the gradient of the penalty is ``2 * lambda_ * w``.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        lambda_: float, the regularisation parameter.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, the number of gradient descent steps to perform.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the weight vector after max_iters steps.
        loss: the average negative log-likelihood at w, without the penalty.
    """
    w = np.asarray(initial_w, dtype=float)
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * gradient
    return w, compute_logistic_loss(y, tx, w)
