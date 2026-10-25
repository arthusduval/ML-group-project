# CS-433 Project 1 — Predicting coronary heart disease from the BRFSS survey

Binary classification of MICHD (myocardial infarction or coronary heart
disease) from the 2015 Behavioral Risk Factor Surveillance System telephone
survey: 328 135 training respondents, 321 raw columns, 8.8 % positives. Only
NumPy and the Python standard library are used.

Best cross-validated score: **F1 = 0.441**, accuracy = 0.877 (5-fold,
out-of-fold predictions).

## Reproducing the submission

```bash
python run.py          # ~2 minutes, ~8 GB of memory, writes submission.csv
```

`run.py` loads the data, fits the preprocessing, cross-validates to select the
decision threshold, refits on the whole training set and writes
`submission.csv`. It is fully deterministic: the fold assignment uses a fixed
seed and the model is fitted by a deterministic Newton solver.

The data is expected in `data/` (`x_train.csv`, `y_train.csv`, `x_test.csv`),
which in this repository is a symbolic link to the folder where the CSV files
were downloaded. The first run parses the 300 MB of CSV in about 40 seconds and
caches the parsed arrays in `cache/raw_data.npz`; later runs load in under a
second.

## Files

| file | contents |
| --- | --- |
| `implementations.py` | the six methods required by the project statement, plus the losses and gradients they share |
| `helpers.py` | CSV loading (with caching) and submission writing |
| `preprocessing.py` | cleaning of the BRFSS codes, encoding, feature construction, removal of linearly dependent columns |
| `model_selection.py` | F1 / accuracy, k-fold cross-validation, threshold selection, the Newton solver we use |
| `eda.py` | exploratory analysis: prints and saves a per-column summary |
| `experiments.py` | the tables of the report: method comparison and ablation study |
| `run.py` | end-to-end reproduction of the submission |
| `report/` | the LaTeX report |
| `figures/` | results produced by `eda.py` and `experiments.py` |

## The six required methods

All of them live in `implementations.py` and return `(w, loss)`, where `loss`
is evaluated at the returned `w` and excludes the penalty term for the
regularised variants.

| function | method |
| --- | --- |
| `mean_squared_error_gd` | linear regression, gradient descent |
| `mean_squared_error_sgd` | linear regression, stochastic gradient descent (batch size 1) |
| `least_squares` | least squares, normal equations |
| `ridge_regression` | ridge regression, normal equations |
| `logistic_regression` | logistic regression, gradient descent |
| `reg_logistic_regression` | L2-regularised logistic regression, gradient descent |

Linear systems are solved with `np.linalg.solve`; no matrix is ever inverted
explicitly.

## Pipeline

1. **Cleaning.** The survey reserves codes for non-answers (`7`/`9`, `77`/`99`,
   `777`/`999`) and encodes some quantities as compound codes (`2xx` = xx times
   per month, weight in pounds or in kilograms offset by 9000, height as
   `ftii`). These are decoded into comparable units and the reserved codes are
   turned into missing values.
2. **Encoding.** Columns with few distinct values are one-hot encoded against
   their most frequent level; the others are winsorised at the 1st and 99th
   percentiles, median-imputed and standardised. A "was missing" indicator is
   added for every column that is missing more than 2 % of the time — in a
   survey, a non-answer is itself informative.
3. **Features.** The square of every numerical feature is appended, so that a
   linear model can represent the U-shaped effect of age or body mass index.
4. **Conditioning.** One-hot encoding creates exact linear dependencies, which
   make `X^T X` singular (rank 344 of 415 columns on the default pipeline).
   Redundant columns are removed by an incremental Cholesky factorisation of
   the Gram matrix, which restores a unique solution without changing the span
   of the features.
5. **Model.** L2-regularised logistic regression (`lambda = 1e-4`) fitted by
   Newton's method, which converges in about ten iterations where gradient
   descent still has not converged after several hundred.
6. **Decision threshold.** With 8.8 % positives, the default threshold predicts
   almost nobody positive. The threshold is selected on out-of-fold scores to
   maximise F1. This single step is worth more than every other choice
   combined: F1 goes from 0.23 to 0.44.

## Results

`python experiments.py` regenerates the tables of the report into
`figures/`. See `report/report.pdf` for the discussion.

## Tests

The public grading tests pass from the course repository:

```bash
cd <ML_course>/projects/project1/grading_tests
pytest --github_link <path-to-this-repository> .
```
