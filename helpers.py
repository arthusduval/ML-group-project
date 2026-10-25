"""Data loading and submission helpers for Project 1.

``load_csv_data`` and ``create_csv_submission`` keep the signatures provided
with the project statement. ``load_csv_data`` additionally caches the parsed
arrays as ``.npy`` files, because parsing the 300 MB of CSV takes about 30
seconds while reloading the cache takes well under a second.
"""

import os

import numpy as np


def load_csv_data(data_path, sub_sample=False, cache_dir="cache"):
    """Load the raw training and test sets.

    Args:
        data_path: path of the folder containing x_train.csv, y_train.csv and
            x_test.csv.
        sub_sample: if True, only keep every 50th training sample.
        cache_dir: folder where the parsed arrays are cached, or None to
            disable caching.

    Returns:
        x_train: numpy array of shape (N, D), the training features.
        x_test: numpy array of shape (N_test, D), the test features.
        y_train: numpy array of shape (N,), the labels in {-1, 1}.
        train_ids: numpy array of shape (N,), the training sample ids.
        test_ids: numpy array of shape (N_test,), the test sample ids.
    """
    cache_file = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "raw_data.npz")
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            arrays = (
                cached["x_train"],
                cached["x_test"],
                cached["y_train"],
                cached["train_ids"],
                cached["test_ids"],
            )
            return _maybe_sub_sample(arrays, sub_sample)

    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    # The first column of the feature files is the sample id, not a feature.
    train_ids = x_train[:, 0].astype(int)
    test_ids = x_test[:, 0].astype(int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    if cache_file is not None:
        np.savez_compressed(
            cache_file,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            train_ids=train_ids,
            test_ids=test_ids,
        )

    arrays = (x_train, x_test, y_train, train_ids, test_ids)
    return _maybe_sub_sample(arrays, sub_sample)


def _maybe_sub_sample(arrays, sub_sample):
    """Keep every 50th training sample when sub_sample is True."""
    x_train, x_test, y_train, train_ids, test_ids = arrays
    if sub_sample:
        x_train = x_train[::50]
        y_train = y_train[::50]
        train_ids = train_ids[::50]
    return x_train, x_test, y_train, train_ids, test_ids


def load_feature_names(data_path):
    """Read the feature names from the header of x_train.csv (without the id).

    Args:
        data_path: path of the folder containing x_train.csv.

    Returns:
        numpy array of shape (D,) containing the column names.
    """
    with open(os.path.join(data_path, "x_train.csv")) as handle:
        header = handle.readline().strip().split(",")
    return np.array(header[1:])


def create_csv_submission(ids, y_pred, name):
    """Write a submission file in the format expected by the competition.

    Args:
        ids: iterable of the test sample ids.
        y_pred: iterable of the predicted labels, in {-1, 1}.
        name: path of the file to write.
    """
    if not np.all(np.isin(y_pred, [-1, 1])):
        raise ValueError("y_pred can only contain values -1 or 1")

    with open(name, "w") as csv_file:
        csv_file.write("Id,Prediction\n")
        for r1, r2 in zip(ids, y_pred):
            csv_file.write(f"{int(r1)},{int(r2)}\n")
