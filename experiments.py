"""Experiments backing the report: baselines and ablation study.

Two tables are produced.

``methods``: the six methods required by the project statement, plus the
variant we selected, all evaluated with the same 5-fold cross-validation and
the same feature pipeline.

``ablation``: the effect of each preprocessing decision, measured by removing
one step at a time from the full pipeline.

The preprocessing is re-estimated inside every training fold, so that no
statistic computed on a validation fold ever reaches the model.
"""

import csv
import time

import numpy as np

from helpers import load_csv_data, load_feature_names
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from model_selection import (
    class_weights,
    cross_validate,
    reg_logistic_regression_newton,
)
from preprocessing import add_bias, fit_preprocessor, independent_columns, transform

DATA_PATH = "data"
K_FOLD = 5
SEED = 1

# Step sizes chosen with the stability bound gamma < 2 / lambda_max(Hessian),
# estimated on the design matrix (see report).
# Largest stable step sizes measured on the final design matrix:
#   lambda_max(X^T X / N) = 170.2  ->  gamma < 2 / 170.2 = 0.0117 for the MSE,
#   max_n ||x_n||^2 = 13753        ->  gamma < 1.45e-4 for a batch of one.
# The logistic bound (4x looser) is very conservative in practice because
# sigma (1 - sigma) is far below 1/4 on an imbalanced data set.
GAMMA_MSE = 0.01
GAMMA_SGD = 1e-4
GAMMA_LOGISTIC = 0.2
MAX_ITERS_GD = 300
MAX_ITERS_SGD = 50000
LAMBDA_RIDGE = 1e-6
LAMBDA_LOGISTIC = 1e-4

# The feature pipeline selected in the report, shared by every method.
PIPELINE = {"max_missing": 1.0, "square_numerical": True}


def make_design_matrix(x_train, x_val, names, full_rank=True, **prep_kwargs):
    """Fit the preprocessing on the training fold and apply it to both folds.

    Args:
        x_train: numpy array of shape (N, D), raw training features.
        x_val: numpy array of shape (M, D), raw validation features.
        names: numpy array of shape (D,), the column names.
        full_rank: if True, linearly dependent columns are removed.
        prep_kwargs: forwarded to ``fit_preprocessor``.

    Returns:
        tx_train, tx_val: the two design matrices, with a bias column.
    """
    params = fit_preprocessor(x_train, names, **prep_kwargs)
    tx_train = add_bias(transform(x_train, params))
    tx_val = add_bias(transform(x_val, params))
    if full_rank:
        keep = independent_columns(tx_train)
        tx_train, tx_val = tx_train[:, keep], tx_val[:, keep]
    return tx_train, tx_val


def make_fit_predict(method, names, prep_kwargs=None, **model_kwargs):
    """Build the ``fit_predict`` callable expected by ``cross_validate``.

    Args:
        method: name of the method to run.
        names: numpy array of shape (D,), the raw column names.
        prep_kwargs: options forwarded to ``fit_preprocessor``.
        model_kwargs: options of the model (lambda_, gamma, ...).

    Returns:
        A callable ``(x_train, y_train, x_val) -> (scores_train, scores_val)``.
    """
    prep_kwargs = prep_kwargs or {}

    def fit_predict(x_train, y_train, x_val):
        tx_train, tx_val = make_design_matrix(x_train, x_val, names, **prep_kwargs)
        initial_w = np.zeros(tx_train.shape[1])

        # The regression methods are trained on labels in {-1, 1} and the
        # logistic ones on labels in {0, 1}, as required by the statement.
        y_regression = y_train.astype(float)
        y_logistic = (y_train == 1).astype(float)

        if method == "least_squares":
            w, _ = least_squares(y_regression, tx_train)
        elif method == "ridge_regression":
            w, _ = ridge_regression(y_regression, tx_train, model_kwargs["lambda_"])
        elif method == "mean_squared_error_gd":
            w, _ = mean_squared_error_gd(
                y_regression, tx_train, initial_w, MAX_ITERS_GD, model_kwargs["gamma"]
            )
        elif method == "mean_squared_error_sgd":
            w, _ = mean_squared_error_sgd(
                y_regression, tx_train, initial_w, MAX_ITERS_SGD, model_kwargs["gamma"]
            )
        elif method == "logistic_regression":
            w, _ = logistic_regression(
                y_logistic, tx_train, initial_w, MAX_ITERS_GD, model_kwargs["gamma"]
            )
        elif method == "reg_logistic_regression":
            w, _ = reg_logistic_regression(
                y_logistic,
                tx_train,
                model_kwargs["lambda_"],
                initial_w,
                MAX_ITERS_GD,
                model_kwargs["gamma"],
            )
        elif method == "newton":
            weights = class_weights(y_logistic, model_kwargs.get("balanced", False))
            w, _ = reg_logistic_regression_newton(
                y_logistic,
                tx_train,
                model_kwargs["lambda_"],
                initial_w,
                weights=weights,
            )
        else:
            raise ValueError(f"unknown method {method}")

        return tx_train.dot(w), tx_val.dot(w)

    return fit_predict


def run_table(rows, x, y, names, k_fold=K_FOLD, tune_threshold=True):
    """Cross-validate a list of configurations and print the result table.

    Args:
        rows: list of ``(label, method, prep_kwargs, model_kwargs)`` tuples.
        x: numpy array of shape (N, D), the raw features.
        y: numpy array of shape (N,), labels in {-1, 1}.
        names: numpy array of shape (D,), the column names.
        k_fold: number of cross-validation folds.
        tune_threshold: whether the decision threshold is tuned on the
            training fold.

    Returns:
        A list of dictionaries, one per configuration.
    """
    results = []
    header = f"{'configuration':<38}{'F1':>16}{'accuracy':>16}{'time':>9}"
    print(header)
    print("-" * len(header))

    for label, method, prep_kwargs, model_kwargs in rows:
        start = time.time()
        try:
            fit_predict = make_fit_predict(method, names, prep_kwargs, **model_kwargs)
            scores = cross_validate(
                x,
                y,
                fit_predict,
                k_fold=k_fold,
                seed=SEED,
                tune_threshold=tune_threshold,
            )
        except np.linalg.LinAlgError as error:
            print(f"{label:<38}{'singular matrix: ' + str(error):>32}")
            results.append({"label": label, "failed": str(error)})
            continue

        elapsed = time.time() - start
        scores["label"] = label
        scores["time"] = elapsed
        results.append(scores)
        print(
            f"{label:<38}"
            f"{scores['f1_mean']:.4f} +- {scores['f1_std']:.4f}"
            f"{scores['accuracy_mean']:>10.4f} +- {scores['accuracy_std']:.4f}"
            f"{elapsed:>8.0f}s"
        )
    return results


def write_csv(results, path):
    """Write a result table to a CSV file.

    Args:
        results: the list returned by ``run_table``.
        path: destination file.
    """
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "configuration",
                "f1_mean",
                "f1_std",
                "accuracy_mean",
                "accuracy_std",
                "time_s",
            ]
        )
        for row in results:
            if "failed" in row:
                writer.writerow([row["label"], "", "", "", "", ""])
                continue
            writer.writerow(
                [
                    row["label"],
                    f"{row['f1_mean']:.4f}",
                    f"{row['f1_std']:.4f}",
                    f"{row['accuracy_mean']:.4f}",
                    f"{row['accuracy_std']:.4f}",
                    f"{row['time']:.1f}",
                ]
            )


def main():
    x, _, y, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)
    print(
        f"{x.shape[0]} samples, {x.shape[1]} raw features, "
        f"positive rate {(y == 1).mean():.4f}\n"
    )

    print("=== Table 1: the six required methods (5-fold CV, tuned threshold) ===")
    methods = [
        ("least squares (normal equations)", "least_squares", PIPELINE, {}),
        ("ridge regression", "ridge_regression", PIPELINE, {"lambda_": LAMBDA_RIDGE}),
        (
            "MSE gradient descent",
            "mean_squared_error_gd",
            PIPELINE,
            {"gamma": GAMMA_MSE},
        ),
        ("MSE stochastic GD", "mean_squared_error_sgd", PIPELINE, {"gamma": GAMMA_SGD}),
        (
            "logistic regression (GD)",
            "logistic_regression",
            PIPELINE,
            {"gamma": GAMMA_LOGISTIC},
        ),
        (
            "regularised logistic (GD)",
            "reg_logistic_regression",
            PIPELINE,
            {"lambda_": LAMBDA_LOGISTIC, "gamma": GAMMA_LOGISTIC},
        ),
        (
            "regularised logistic (Newton, ours)",
            "newton",
            PIPELINE,
            {"lambda_": LAMBDA_LOGISTIC},
        ),
    ]
    results = run_table(methods, x, y, names)
    write_csv(results, "figures/results_methods.csv")

    print("\n=== Table 2: the decision threshold ===")
    threshold_rows = [
        (
            "Newton, threshold fixed at 0",
            "newton",
            PIPELINE,
            {"lambda_": LAMBDA_LOGISTIC},
        ),
    ]
    fixed = run_table(threshold_rows, x, y, names, tune_threshold=False)
    write_csv(fixed, "figures/results_threshold.csv")

    print("\n=== Table 3: ablation of the pipeline (Newton, 3-fold CV) ===")
    without = lambda **kwargs: {**PIPELINE, **kwargs}
    ablation = [
        ("full pipeline", "newton", PIPELINE, {"lambda_": LAMBDA_LOGISTIC}),
        (
            "without quadratic terms",
            "newton",
            without(square_numerical=False),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "without one-hot encoding",
            "newton",
            without(max_categories=0),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "without missingness indicators",
            "newton",
            without(missing_indicator_threshold=1.1),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "without reserved-code masking",
            "newton",
            without(mask_codes=False),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "without compound decoding",
            "newton",
            without(decode_compound=False),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "dropping columns over 50% missing",
            "newton",
            without(max_missing=0.5),
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "no cleaning at all (raw codes)",
            "newton",
            {
                "max_missing": 1.0,
                "mask_codes": False,
                "decode_compound": False,
                "max_categories": 0,
                "missing_indicator_threshold": 1.1,
            },
            {"lambda_": LAMBDA_LOGISTIC},
        ),
        (
            "with balanced class weights",
            "newton",
            PIPELINE,
            {"lambda_": LAMBDA_LOGISTIC, "balanced": True},
        ),
    ]
    results = run_table(ablation, x, y, names, k_fold=3)
    write_csv(results, "figures/results_ablation.csv")


if __name__ == "__main__":
    main()
