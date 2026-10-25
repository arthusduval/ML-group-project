"""Cleaning and feature engineering for the BRFSS 2015 data set.

The raw data is a telephone survey: almost every column is an integer code
rather than a number, roughly 45% of the entries are missing, and the survey
uses reserved codes ("7 = don't know", "9 = refused", "88 = none", ...) that
are numerically meaningless. The pipeline below turns this into a purely
numerical design matrix:

1. administrative columns (identifiers, interview dates, survey design
   weights) are dropped, since they carry no medical information and the
   sampling weights leak the survey design into the model;
2. reserved codes are mapped to "missing" and the columns that encode a
   frequency as a compound code (``2xx`` = xx times per week, ...) are decoded
   into a single comparable unit;
3. columns that are missing too often are dropped;
4. columns with few distinct values are treated as categorical and one-hot
   encoded (one indicator per level, the most frequent level being used as the
   reference), while the remaining columns are treated as numerical,
   winsorised, median-imputed and standardised;
5. an explicit "was missing" indicator is added for every column whose
   missingness is informative, since in a survey a non-answer is itself a
   signal.

All the statistics used by the transformation (levels, medians, quantiles,
means, standard deviations) are estimated on the training set only and stored
in a dictionary, so that the exact same transformation can be replayed on the
test set without leaking any information.
"""

import numpy as np

# Identifiers, interview dates and survey design weights: no medical content.
ADMIN_COLUMNS = [
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "FMONTH",
    "SEQNO",
    "_PSU",
    "DISPCODE",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_DUALCOR",
    "_LLCPWT",
    "_CLLCPWT",
    "_LLCPWT2",
    "HIVTSTD3",
    "FLSHTMY2",
    "CTELENUM",
    "CTELNUM1",
    "PVTRESD1",
    "PVTRESD2",
    "CELLFON2",
    "CELLFON3",
    "LANDLINE",
]

# Columns encoding "how often", with the BRFSS compound convention
#   1xx -> xx times per week, 2xx -> xx times per month, 3xx -> xx times per year.
# They are decoded into a single unit: number of times per week.
WEEKLY_FREQUENCY_COLUMNS = [
    "ALCDAY5",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXEROFT1",
    "EXEROFT2",
    "STRENGTH",
]

# Columns counting a number of days over the last 30, where 88 means "none".
DAY_COUNT_COLUMNS = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]

# Columns counting a number of events, where 88 means "none".
NONE_IS_88_COLUMNS = ["CHILDREN", "DRNK3GE5", "AVEDRNK2", "MAXDRNKS", "ALCDAY5"]

# Columns whose integer codes are labels, not quantities, whatever their
# cardinality: they must be one-hot encoded rather than treated as numbers.
FORCE_CATEGORICAL_COLUMNS = ["_STATE"]


def decode_weekly_frequency(column):
    """Decode a BRFSS compound frequency code into times per week.

    The convention is ``1xx`` = xx times per week, ``2xx`` = xx times per
    month and ``3xx`` = xx times per year; ``555`` means "never" and
    ``777``/``999`` mean "don't know"/"refused".

    Args:
        column: numpy array of shape (N,), the raw codes.

    Returns:
        numpy array of shape (N,), the frequency in times per week, with NaN
        where the answer was missing or refused.
    """
    out = np.full(column.shape, np.nan)

    never = np.isin(column, [555, 888])
    out[never] = 0.0

    per_week = (column >= 101) & (column <= 199)
    out[per_week] = column[per_week] - 100

    per_month = (column >= 201) & (column <= 299)
    out[per_month] = (column[per_month] - 200) * 12.0 / 52.0

    per_year = (column >= 301) & (column <= 399)
    out[per_year] = (column[per_year] - 300) / 52.0

    return out


def decode_weight_kg(column):
    """Decode the BRFSS WEIGHT2 column into kilograms.

    Values between 50 and 776 are pounds, values between 9023 and 9352 are
    kilograms offset by 9000, and 7777/9999 mean "don't know"/"refused".

    Args:
        column: numpy array of shape (N,), the raw codes.

    Returns:
        numpy array of shape (N,), the weight in kg, NaN where unknown.
    """
    out = np.full(column.shape, np.nan)
    pounds = (column >= 50) & (column <= 776)
    out[pounds] = column[pounds] * 0.4536
    kilograms = (column >= 9000) & (column <= 9998)
    out[kilograms] = column[kilograms] - 9000
    return out


def decode_height_m(column):
    """Decode the BRFSS HEIGHT3 column into metres.

    Values between 200 and 711 are feet and inches (``ftii``), values between
    9061 and 9998 are centimetres offset by 9000.

    Args:
        column: numpy array of shape (N,), the raw codes.

    Returns:
        numpy array of shape (N,), the height in metres, NaN where unknown.
    """
    out = np.full(column.shape, np.nan)
    imperial = (column >= 200) & (column <= 711)
    feet = np.floor(column[imperial] / 100.0)
    inches = column[imperial] - 100 * feet
    out[imperial] = (12.0 * feet + inches) * 0.0254
    metric = (column >= 9000) & (column <= 9998)
    out[metric] = (column[metric] - 9000) / 100.0
    return out


def mask_reserved_codes(column):
    """Replace the BRFSS "don't know" / "refused" codes by NaN.

    The survey reserves the codes 7/9, 77/99, 777/999 and 7777/9999 (depending
    on the width of the field) for non-answers. The width is inferred from the
    largest value actually observed in the column.

    Args:
        column: numpy array of shape (N,), a raw column.

    Returns:
        numpy array of shape (N,), with the reserved codes replaced by NaN.
    """
    out = column.astype(float).copy()
    observed = out[~np.isnan(out)]
    if observed.size == 0:
        return out

    largest = observed.max()
    if largest <= 9:
        reserved = [7, 9]
    elif largest <= 99:
        reserved = [77, 99]
    elif largest <= 999:
        reserved = [777, 999]
    else:
        reserved = [7777, 9999, 99900, 999999]

    out[np.isin(out, reserved)] = np.nan
    return out


def clean_raw_columns(x, names, decode_compound=True, mask_codes=True):
    """Apply the column-specific decoding and the reserved-code masking.

    Args:
        x: numpy array of shape (N, D), the raw features.
        names: numpy array of shape (D,), the column names.
        decode_compound: if False, the compound frequency, weight and height
            codes are left as raw integers (used for the ablation study).
        mask_codes: if False, the reserved "don't know" / "refused" codes are
            kept as numbers (used for the ablation study).

    Returns:
        numpy array of shape (N, D) of floats, cleaned but not yet encoded.
    """
    cleaned = x.astype(float).copy()
    if not decode_compound and not mask_codes:
        return cleaned

    name_to_index = {name: index for index, name in enumerate(names)}
    if not decode_compound:
        for index, name in enumerate(names):
            cleaned[:, index] = mask_reserved_codes(cleaned[:, index])
        return cleaned

    for name in WEEKLY_FREQUENCY_COLUMNS:
        if name in name_to_index:
            index = name_to_index[name]
            cleaned[:, index] = decode_weekly_frequency(cleaned[:, index])

    for name in DAY_COUNT_COLUMNS + NONE_IS_88_COLUMNS:
        if name in name_to_index:
            index = name_to_index[name]
            column = cleaned[:, index]
            # 88 means "none" in these columns, not the number 88.
            column[column == 88] = 0.0
            cleaned[:, index] = column

    if "WEIGHT2" in name_to_index:
        index = name_to_index["WEIGHT2"]
        cleaned[:, index] = decode_weight_kg(cleaned[:, index])
    if "HEIGHT3" in name_to_index:
        index = name_to_index["HEIGHT3"]
        cleaned[:, index] = decode_height_m(cleaned[:, index])

    if mask_codes:
        decoded = set(WEEKLY_FREQUENCY_COLUMNS) | {"WEIGHT2", "HEIGHT3"}
        for index, name in enumerate(names):
            if name not in decoded:
                cleaned[:, index] = mask_reserved_codes(cleaned[:, index])

    return cleaned


def fit_preprocessor(
    x,
    names,
    max_missing=0.5,
    max_categories=12,
    min_level_count=100,
    missing_indicator_threshold=0.02,
    decode_compound=True,
    mask_codes=True,
    square_numerical=False,
):
    """Estimate every statistic needed to build the design matrix.

    Args:
        x: numpy array of shape (N, D), the raw training features.
        names: numpy array of shape (D,), the column names.
        max_missing: columns missing more often than this are dropped.
        max_categories: columns with at most this many distinct values are
            treated as categorical.
        min_level_count: categorical levels rarer than this are merged into
            the reference level.
        missing_indicator_threshold: a "was missing" indicator is created for
            columns whose missing rate exceeds this value.
        decode_compound: whether to decode the compound BRFSS codes.
        mask_codes: whether to treat the reserved codes as missing values.
        square_numerical: whether to append the square of every standardised
            numerical feature, which lets a linear model capture the U-shaped
            effects of age or body mass index.

    Returns:
        A dictionary holding the fitted preprocessing parameters.
    """
    cleaned = clean_raw_columns(x, names, decode_compound, mask_codes)

    keep = np.array([name not in ADMIN_COLUMNS for name in names])
    missing_rate = np.isnan(cleaned).mean(axis=0)
    keep &= missing_rate <= max_missing

    categorical, numerical = [], []
    for index in np.flatnonzero(keep):
        column = cleaned[:, index]
        values = column[~np.isnan(column)]
        distinct = np.unique(values)
        if distinct.size <= 1:
            continue  # constant column, no information
        if distinct.size <= max_categories or names[index] in FORCE_CATEGORICAL_COLUMNS:
            counts = np.array([(values == level).sum() for level in distinct])
            frequent = distinct[counts >= min_level_count]
            if frequent.size <= 1:
                continue
            # The most frequent level is the reference and gets no indicator.
            reference = frequent[np.argmax(counts[counts >= min_level_count])]
            levels = [level for level in frequent if level != reference]
            categorical.append({"index": index, "levels": levels})
        else:
            finite = values
            low, high = np.percentile(finite, [1.0, 99.0])
            clipped = np.clip(finite, low, high)
            mean = clipped.mean()
            std = clipped.std()
            numerical.append(
                {
                    "index": index,
                    "low": low,
                    "high": high,
                    "median": np.median(finite),
                    "mean": mean,
                    "std": std if std > 0 else 1.0,
                }
            )

    indicators = [
        index
        for index in np.flatnonzero(keep)
        if missing_rate[index] > missing_indicator_threshold
    ]

    return {
        "names": names,
        "categorical": categorical,
        "numerical": numerical,
        "indicators": indicators,
        "decode_compound": decode_compound,
        "mask_codes": mask_codes,
        "square_numerical": square_numerical,
    }


def transform(x, params):
    """Build the design matrix of a data set with fitted parameters.

    Args:
        x: numpy array of shape (N, D), raw features.
        params: the dictionary returned by ``fit_preprocessor``.

    Returns:
        numpy array of shape (N, P), the design matrix without bias column.
    """
    cleaned = clean_raw_columns(
        x, params["names"], params["decode_compound"], params["mask_codes"]
    )
    blocks = []

    numerical_blocks = []
    for spec in params["numerical"]:
        column = cleaned[:, spec["index"]]
        filled = np.where(np.isnan(column), spec["median"], column)
        filled = np.clip(filled, spec["low"], spec["high"])
        numerical_blocks.append(((filled - spec["mean"]) / spec["std"])[:, None])
    blocks.extend(numerical_blocks)
    if params["square_numerical"]:
        blocks.extend([block**2 for block in numerical_blocks])

    for spec in params["categorical"]:
        column = cleaned[:, spec["index"]]
        for level in spec["levels"]:
            blocks.append((column == level).astype(float)[:, None])

    for index in params["indicators"]:
        blocks.append(np.isnan(cleaned[:, index]).astype(float)[:, None])

    return np.hstack(blocks)


def feature_names(params):
    """Return the name of every column of the design matrix.

    Args:
        params: the dictionary returned by ``fit_preprocessor``.

    Returns:
        A list of strings, one per column of the matrix built by ``transform``.
    """
    names = params["names"]
    out = [f"{names[spec['index']]}" for spec in params["numerical"]]
    if params["square_numerical"]:
        out += [f"{names[spec['index']]}^2" for spec in params["numerical"]]
    for spec in params["categorical"]:
        out += [f"{names[spec['index']]}={level:g}" for level in spec["levels"]]
    out += [f"{names[index]}:missing" for index in params["indicators"]]
    return out


def add_bias(x):
    """Prepend a column of ones to a feature matrix.

    Args:
        x: numpy array of shape (N, P).

    Returns:
        numpy array of shape (N, P + 1).
    """
    return np.hstack([np.ones((x.shape[0], 1)), x])


def independent_columns(x, tol=1e-9):
    """Select a maximal subset of linearly independent columns of ``x``.

    One-hot encoding a survey produces exact linear dependencies (a level that
    never co-occurs with another, a missingness indicator that is already
    implied by the level dummies, ...). The Gram matrix ``x.T @ x`` is then
    singular and the normal equations of ``least_squares`` have no unique
    solution. This routine runs an incremental Cholesky factorisation of the
    Gram matrix and keeps a column only when it adds a direction that the
    already selected columns cannot represent, which restores full rank while
    changing nothing to the span of the features.

    Args:
        x: numpy array of shape (N, P), the design matrix.
        tol: relative threshold below which a column is declared redundant.

    Returns:
        numpy array of the indices of the columns to keep, in increasing order.
    """
    gram = x.T.dot(x)
    keep = []
    factor = np.zeros((0, 0))

    for column in range(gram.shape[0]):
        if len(keep) == 0:
            projection = np.zeros(0)
        else:
            # Solve L u = gram[keep, column] with L lower triangular.
            projection = np.linalg.solve(factor, gram[keep, column])

        pivot = gram[column, column] - projection.dot(projection)
        if pivot <= tol * max(gram[column, column], 1.0):
            continue  # column is a linear combination of the selected ones

        size = len(keep)
        updated = np.zeros((size + 1, size + 1))
        updated[:size, :size] = factor
        updated[size, :size] = projection
        updated[size, size] = np.sqrt(pivot)
        factor = updated
        keep.append(column)

    return np.array(keep)
