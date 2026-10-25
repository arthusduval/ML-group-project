"""Metrics, cross-validation and the model variants used on top of the
six required methods.

The competition is scored with the F1 score on a data set where only 8.8% of
the patients are positive. Two consequences drive everything in this file:

* accuracy is a useless metric here (always predicting "healthy" already gives
  91.2%), so every model is selected on F1;
* the decision threshold is a hyperparameter of its own. A model trained on an
  imbalanced set puts almost every predicted probability below 0.5, so the
  threshold has to be tuned on validation data rather than left at its default.
"""

import numpy as np

from implementations import compute_logistic_loss, sigmoid


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def accuracy(y_true, y_pred):
    """Fraction of correctly classified samples.

    Args:
        y_true: numpy array of shape (N,), the true labels.
        y_pred: numpy array of shape (N,), the predicted labels.

    Returns:
        The accuracy, a float in [0, 1].
    """
    return np.mean(y_true == y_pred)


def f1_score(y_true, y_pred, positive=1):
    """F1 score of the positive class.

    Args:
        y_true: numpy array of shape (N,), the true labels.
        y_pred: numpy array of shape (N,), the predicted labels.
        positive: the value encoding the positive class.

    Returns:
        The F1 score, a float in [0, 1].
    """
    true_positive = np.sum((y_pred == positive) & (y_true == positive))
    false_positive = np.sum((y_pred == positive) & (y_true != positive))
    false_negative = np.sum((y_pred != positive) & (y_true == positive))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return 2.0 * true_positive / denominator


def best_f1_threshold(y_true, scores):
    """Find the decision threshold on ``scores`` that maximises the F1 score.

    All candidate thresholds are evaluated at once: after sorting the scores in
    decreasing order, the number of true positives above a threshold is a
    cumulative sum, so the whole F1 curve costs one sort.

    Args:
        y_true: numpy array of shape (N,), labels in {-1, 1}.
        scores: numpy array of shape (N,), the continuous model output.

    Returns:
        threshold: the score above which a sample is predicted positive.
        f1: the F1 score reached at that threshold.
    """
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    positives = (y_true[order] == 1).astype(float)

    true_positive = np.cumsum(positives)
    predicted_positive = np.arange(1, scores.size + 1)
    total_positive = positives.sum()

    f1_curve = 2.0 * true_positive / (predicted_positive + total_positive)

    # Only thresholds that fall between two distinct scores are reachable.
    reachable = np.ones(scores.size, dtype=bool)
    reachable[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    f1_curve = np.where(reachable, f1_curve, -1.0)

    best = int(np.argmax(f1_curve))
    return sorted_scores[best], f1_curve[best]


def predict_labels(scores, threshold=0.0):
    """Turn continuous scores into {-1, 1} labels.

    Args:
        scores: numpy array of shape (N,), the continuous model output.
        threshold: samples with a score above it are predicted positive.

    Returns:
        numpy array of shape (N,) with values in {-1, 1}.
    """
    return np.where(scores > threshold, 1, -1)


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def build_k_indices(num_rows, k_fold, seed=1):
    """Split the row indices into k random folds of equal size.

    Args:
        num_rows: int, the number of samples.
        k_fold: int, the number of folds.
        seed: int, the seed of the random permutation.

    Returns:
        numpy array of shape (k_fold, num_rows // k_fold) of row indices.
    """
    interval = num_rows // k_fold
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_rows)
    return np.array([indices[k * interval : (k + 1) * interval] for k in range(k_fold)])


def cross_validate(x, y, fit_predict, k_fold=5, seed=1, tune_threshold=True):
    """Estimate the F1 and accuracy of a model by k-fold cross-validation.

    Args:
        x: numpy array of shape (N, P), the design matrix (with bias column).
        y: numpy array of shape (N,), labels in {-1, 1}.
        fit_predict: callable ``(x_train, y_train, x_val) -> scores``, which
            trains a model and returns the continuous score of the validation
            samples.
        k_fold: int, the number of folds.
        seed: int, the seed used to build the folds.
        tune_threshold: if True the decision threshold is chosen on the
            training fold, otherwise 0 is used.

    Returns:
        A dictionary with the mean and standard deviation of the F1 score and
        the accuracy over the folds, and the mean selected threshold.
    """
    k_indices = build_k_indices(x.shape[0], k_fold, seed)
    scores_f1, scores_accuracy, thresholds = [], [], []

    for k in range(k_fold):
        validation = k_indices[k]
        training = np.concatenate([k_indices[j] for j in range(k_fold) if j != k])

        train_scores, validation_scores = fit_predict(
            x[training], y[training], x[validation]
        )

        if tune_threshold:
            threshold, _ = best_f1_threshold(y[training], train_scores)
        else:
            threshold = 0.0

        prediction = predict_labels(validation_scores, threshold)
        scores_f1.append(f1_score(y[validation], prediction))
        scores_accuracy.append(accuracy(y[validation], prediction))
        thresholds.append(threshold)

    return {
        "f1_mean": float(np.mean(scores_f1)),
        "f1_std": float(np.std(scores_f1)),
        "accuracy_mean": float(np.mean(scores_accuracy)),
        "accuracy_std": float(np.std(scores_accuracy)),
        "threshold_mean": float(np.mean(thresholds)),
    }


# ---------------------------------------------------------------------------
# Model variants built on top of the required methods
# ---------------------------------------------------------------------------


def class_weights(y, balanced=True):
    """Per-sample weights that rebalance the two classes.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        balanced: if False, returns uniform weights.

    Returns:
        numpy array of shape (N,) of weights averaging to 1.
    """
    if not balanced:
        return np.ones_like(y, dtype=float)
    positive_rate = y.mean()
    weights = np.where(y == 1, 0.5 / positive_rate, 0.5 / (1.0 - positive_rate))
    return weights


def reg_logistic_regression_newton(
    y, tx, lambda_, initial_w, max_iters=12, tolerance=1e-8, weights=None
):
    """Regularised logistic regression solved with Newton's method (IRLS).

    Gradient descent needs thousands of iterations on this data set because the
    features have very different curvatures. Newton's method rescales the
    gradient by the Hessian and converges in about ten iterations instead;
    since the design matrix has only a few hundred columns, solving the
    ``(P, P)`` Newton system is cheap compared to one pass over the samples.

    The objective is the (optionally weighted) average negative log-likelihood
    plus ``lambda_ * ||w||^2``.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, P), the design matrix.
        lambda_: float, the regularisation parameter.
        initial_w: numpy array of shape (P,), the initial weight vector.
        max_iters: int, the maximum number of Newton steps.
        tolerance: stop when the weights move less than this in norm.
        weights: optional numpy array of shape (N,), per-sample weights.

    Returns:
        w: numpy array of shape (P,), the fitted weight vector.
        loss: the unweighted, unpenalised logistic loss at w.
    """
    num_samples, num_features = tx.shape
    w = np.asarray(initial_w, dtype=float).copy()
    if weights is None:
        weights = np.ones(num_samples)

    penalty = 2.0 * lambda_ * np.eye(num_features)
    for _ in range(max_iters):
        probability = sigmoid(tx.dot(w))
        gradient = tx.T.dot(weights * (probability - y)) / num_samples
        gradient += 2.0 * lambda_ * w

        # Hessian of the weighted log-likelihood: X^T diag(s (1 - s) c) X / N.
        curvature = weights * probability * (1.0 - probability)
        hessian = tx.T.dot(tx * curvature[:, None]) / num_samples + penalty

        step = np.linalg.solve(hessian, gradient)
        w = w - step
        if np.linalg.norm(step) < tolerance:
            break

    return w, compute_logistic_loss(y, tx, w)
