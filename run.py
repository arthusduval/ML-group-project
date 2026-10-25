"""Reproduce our best submission to the competition.

The script performs the whole pipeline end to end:

1. load the raw CSV files;
2. fit the preprocessing on the training set and build the design matrix;
3. estimate the F1 score and select the decision threshold by 5-fold
   cross-validation, using only out-of-fold predictions;
4. refit the final model on the whole training set;
5. write ``submission.csv``.

Running it takes about two minutes and roughly 8 GB of memory on a laptop.
"""

import time

import numpy as np

from helpers import create_csv_submission, load_csv_data, load_feature_names
from model_selection import (
    accuracy,
    best_f1_threshold,
    build_k_indices,
    f1_score,
    predict_labels,
    reg_logistic_regression_newton,
)
from preprocessing import add_bias, fit_preprocessor, independent_columns, transform

DATA_PATH = "data"
SUBMISSION_PATH = "submission.csv"
SEED = 1
K_FOLD = 5

# Hyperparameters selected by cross-validation, see the report.
MAX_MISSING = 1.0  # keep every column: even nearly empty ones carry signal
SQUARE_NUMERICAL = True  # quadratic effects of age and body mass index
LAMBDA = 1e-4  # L2 penalty of the logistic model
NEWTON_ITERS = 12


def build_features(x_train, x_test, names):
    """Fit the preprocessing on the training set and encode both sets.

    Args:
        x_train: numpy array of shape (N, D), the raw training features.
        x_test: numpy array of shape (M, D), the raw test features.
        names: numpy array of shape (D,), the raw column names.

    Returns:
        tx_train: numpy array of shape (N, P), the training design matrix.
        tx_test: numpy array of shape (M, P), the test design matrix.
    """
    params = fit_preprocessor(
        x_train,
        names,
        max_missing=MAX_MISSING,
        square_numerical=SQUARE_NUMERICAL,
    )
    tx_train = add_bias(transform(x_train, params))
    tx_test = add_bias(transform(x_test, params))

    # One-hot encoding creates exact linear dependencies; removing them keeps
    # the Newton system non-singular without changing the span of the features.
    keep = independent_columns(tx_train)
    return tx_train[:, keep], tx_test[:, keep]


def out_of_fold_scores(tx, y, k_fold=K_FOLD, seed=SEED):
    """Compute an out-of-fold score for every training sample.

    Args:
        tx: numpy array of shape (N, P), the design matrix.
        y: numpy array of shape (N,), labels in {-1, 1}.
        k_fold: number of folds.
        seed: seed of the fold assignment.

    Returns:
        scores: numpy array of shape (N,), NaN for the samples that were left
            out of every fold because N is not divisible by k_fold.
    """
    y_logistic = (y == 1).astype(float)
    k_indices = build_k_indices(tx.shape[0], k_fold, seed)
    scores = np.full(tx.shape[0], np.nan)

    for k in range(k_fold):
        validation = k_indices[k]
        training = np.concatenate([k_indices[j] for j in range(k_fold) if j != k])
        w, _ = reg_logistic_regression_newton(
            y_logistic[training],
            tx[training],
            LAMBDA,
            np.zeros(tx.shape[1]),
            max_iters=NEWTON_ITERS,
        )
        scores[validation] = tx[validation].dot(w)
        print(f"  fold {k + 1}/{k_fold} done")

    return scores


def main():
    start = time.time()

    print("loading the data")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)
    print(f"  {x_train.shape[0]} training samples, {x_train.shape[1]} raw features")

    print("building the features")
    tx_train, tx_test = build_features(x_train, x_test, names)
    print(f"  design matrix: {tx_train.shape[1]} columns")

    print("cross-validating")
    scores = out_of_fold_scores(tx_train, y_train)
    evaluated = ~np.isnan(scores)
    threshold, f1_out_of_fold = best_f1_threshold(y_train[evaluated], scores[evaluated])
    prediction = predict_labels(scores[evaluated], threshold)
    print(f"  out-of-fold F1       : {f1_out_of_fold:.4f}")
    print(f"  out-of-fold accuracy : {accuracy(y_train[evaluated], prediction):.4f}")
    print(f"  selected threshold   : {threshold:.4f}")

    print("fitting the final model on the whole training set")
    w, loss = reg_logistic_regression_newton(
        (y_train == 1).astype(float),
        tx_train,
        LAMBDA,
        np.zeros(tx_train.shape[1]),
        max_iters=NEWTON_ITERS,
    )
    print(f"  training logistic loss: {loss:.5f}")
    print(
        f"  training F1           : "
        f"{f1_score(y_train, predict_labels(tx_train.dot(w), threshold)):.4f}"
    )

    print("writing the submission")
    y_pred = predict_labels(tx_test.dot(w), threshold)
    create_csv_submission(test_ids, y_pred, SUBMISSION_PATH)
    print(f"  {SUBMISSION_PATH}: {(y_pred == 1).mean():.4f} predicted positive")
    print(f"done in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
