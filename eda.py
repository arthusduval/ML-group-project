"""Exploratory data analysis for Project 1.

Running this script prints a compact description of every raw column
(missingness, number of distinct values, range, univariate association with the
target) and writes the table to ``figures/eda_columns.csv`` so that it can be
inspected offline. Nothing here is imported by ``run.py``; the script only
documents how the preprocessing decisions in ``preprocessing.py`` were made.
"""

import numpy as np

from helpers import load_csv_data, load_feature_names

DATA_PATH = "data"


def column_summary(x, y, names):
    """Describe every column of x and its univariate link with y.

    Args:
        x: numpy array of shape (N, D), the raw features.
        y: numpy array of shape (N,), labels in {-1, 1}.
        names: numpy array of shape (D,), the column names.

    Returns:
        A list of dictionaries, one per column.
    """
    positive = y == 1
    rows = []
    for j in range(x.shape[1]):
        column = x[:, j]
        observed = ~np.isnan(column)
        values = column[observed]
        distinct = np.unique(values)

        # Point-biserial correlation, computed on the observed entries only.
        correlation = np.nan
        if values.size > 0 and values.std() > 0:
            target = positive[observed].astype(float)
            if target.std() > 0:
                correlation = np.corrcoef(values, target)[0, 1]

        rows.append(
            {
                "name": names[j],
                "missing": 1.0 - observed.mean(),
                "n_unique": distinct.size,
                "min": values.min() if values.size else np.nan,
                "max": values.max() if values.size else np.nan,
                "corr": correlation,
            }
        )
    return rows


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    print(f"training set: {x_train.shape[0]} samples, {x_train.shape[1]} features")
    print(f"positive rate: {(y_train == 1).mean():.4f}")
    print(f"overall missing rate: {np.isnan(x_train).mean():.4f}\n")

    rows = column_summary(x_train, y_train, names)

    missing = np.array([row["missing"] for row in rows])
    n_unique = np.array([row["n_unique"] for row in rows])
    print("missingness of the columns:")
    for threshold in [0.0, 0.2, 0.5, 0.8, 0.9]:
        print(
            f"  > {threshold:>4.0%} missing: {(missing > threshold).sum():>3d} columns"
        )
    print("\nnumber of distinct values:")
    for threshold in [1, 2, 5, 10, 20, 50]:
        print(
            f"  <= {threshold:>3d} distinct: {(n_unique <= threshold).sum():>3d} columns"
        )

    print("\n30 strongest univariate associations with the target:")
    order = np.argsort(
        [-abs(row["corr"]) if np.isfinite(row["corr"]) else 0 for row in rows]
    )
    print(
        f"{'column':<12}{'corr':>8}{'missing':>10}{'n_unique':>10}{'min':>10}{'max':>10}"
    )
    for index in order[:30]:
        row = rows[index]
        print(
            f"{row['name']:<12}{row['corr']:>8.3f}{row['missing']:>10.2%}"
            f"{row['n_unique']:>10d}{row['min']:>10.1f}{row['max']:>10.1f}"
        )

    with open("figures/eda_columns.csv", "w") as handle:
        handle.write("name,missing,n_unique,min,max,corr\n")
        for row in rows:
            handle.write(
                f"{row['name']},{row['missing']:.6f},{row['n_unique']},"
                f"{row['min']},{row['max']},{row['corr']:.6f}\n"
            )
    print("\nfull table written to figures/eda_columns.csv")


if __name__ == "__main__":
    main()
