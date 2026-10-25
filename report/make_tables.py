"""Turn the CSV files written by ``experiments.py`` into the LaTeX tables.

Run from the repository root::

    python report/make_tables.py

It rewrites the two table bodies of ``report/report.tex`` between the
``% BEGIN <name>`` and ``% END <name>`` markers, so that the numbers in the
report can never drift from the numbers produced by the experiments.
"""

import csv
import re

REPORT = "report/report.tex"

# Short labels for the tables: the CSV keeps descriptive names, the report
# needs names that fit in a two-column layout.
SHORT_LABELS = {
    "least squares (normal equations)": "Least squares (normal eq.)",
    "ridge regression": "Ridge regression",
    "MSE gradient descent": "MSE gradient descent",
    "MSE stochastic GD": "MSE stochastic GD",
    "logistic regression (GD)": "Logistic regression (GD)",
    "regularised logistic (GD)": "Reg. logistic (GD)",
    "regularised logistic (Newton)": "Reg. logistic (Newton)",
    "Newton with the default threshold": "Newton, default threshold",
    "full pipeline": "Full pipeline",
    "without quadratic terms": "$-$ quadratic terms",
    "without one-hot encoding": "$-$ one-hot encoding",
    "without missingness indicators": "$-$ missingness indicators",
    "without reserved-code masking": "$-$ reserved-code masking",
    "without compound decoding": "$-$ compound decoding",
    "dropping columns over 50% missing": "$-$ columns $>50\\%$ missing",
    "no cleaning at all (raw codes)": "No cleaning at all",
    "with balanced class weights": "$+$ balanced class weights",
}


def label(name):
    """Return the short, LaTeX-safe label of a configuration."""
    return SHORT_LABELS.get(name, name.replace("%", "\\%"))


def read_results(path):
    """Read one result file written by ``experiments.py``.

    Args:
        path: path of the CSV file.

    Returns:
        A list of dictionaries with the parsed numeric fields.
    """
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if not row["f1_mean"]:
                rows.append({"configuration": row["configuration"], "failed": True})
                continue
            rows.append(
                {
                    "configuration": row["configuration"],
                    "f1": float(row["f1_mean"]),
                    "f1_std": float(row["f1_std"]),
                    "accuracy": float(row["accuracy_mean"]),
                    "time": float(row["time_s"]),
                }
            )
    return rows


def methods_table(rows):
    """Format the method comparison table."""
    lines = []
    for row in rows:
        name = label(row["configuration"])
        if row.get("failed"):
            lines.append(f"{name} & \\multicolumn{{3}}{{c}}{{singular matrix}} \\\\")
            continue
        lines.append(
            f"{name} & ${row['f1']:.3f} \\pm {row['f1_std']:.3f}$ "
            f"& ${row['accuracy']:.3f}$ & ${row['time']:.0f}$\\,s \\\\"
        )
    return "\n".join(lines)


def ablation_table(rows, reference_label="full pipeline"):
    """Format the ablation table, with the gap to the full pipeline."""
    reference = next(
        row["f1"] for row in rows if row["configuration"] == reference_label
    )
    lines = []
    for row in rows:
        delta = row["f1"] - reference
        gap = "---" if row["configuration"] == reference_label else f"${delta:+.3f}$"
        lines.append(
            f"{label(row['configuration'])} & "
            f"${row['f1']:.3f} \\pm {row['f1_std']:.3f}$ & {gap} \\\\"
        )
    return "\n".join(lines)


def substitute(text, marker, body):
    """Replace the block delimited by the markers of the given name."""
    pattern = re.compile(rf"(% BEGIN {marker}\n).*?(% END {marker})", re.DOTALL)
    if not pattern.search(text):
        raise ValueError(f"markers for {marker} not found in {REPORT}")
    return pattern.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main():
    methods = read_results("figures/results_methods.csv")
    threshold = read_results("figures/results_threshold.csv")
    ablation = read_results("figures/results_ablation.csv")

    text = open(REPORT).read()
    text = substitute(text, "METHODS", methods_table(methods + threshold))
    text = substitute(text, "ABLATION", ablation_table(ablation))
    open(REPORT, "w").write(text)

    print(f"{REPORT} updated")
    for row in methods + threshold + ablation:
        if not row.get("failed"):
            print(f"  {row['configuration']:<40} F1 = {row['f1']:.4f}")


if __name__ == "__main__":
    main()
