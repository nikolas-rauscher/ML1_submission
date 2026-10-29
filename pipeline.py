"""NumPy-only data loading, training-only preprocessing and local metrics."""

import csv
import hashlib
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_features(path, limit=None):
    """Read CSV without pandas; preserve integer IDs and feature ordering."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows, ids = [], []
        for row in reader:
            ids.append(int(row[0]))
            rows.append([float(v) if v else np.nan for v in row[1:]])
            if limit is not None and len(ids) >= limit:
                break
    x = np.asarray(rows, dtype=np.float64)
    return np.asarray(ids, dtype=np.int64), x, header[1:]


def load_labels(path, expected_ids, limit=None):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = []
        for row in reader:
            rows.append((int(row[0]), int(row[1])))
            if limit is not None and len(rows) >= limit:
                break
    a = np.asarray(rows, dtype=np.int64)
    if not np.array_equal(a[:, 0], expected_ids):
        raise ValueError("Training label IDs do not match the feature IDs/order")
    if not np.isin(a[:, 1], [-1, 1]).all():
        raise ValueError("Expected training labels in {-1,1}")
    return a[:, 1]


def stratified_split(y, seed=433, fraction=0.2):
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for label in [-1, 1]:
        idx = rng.permutation(np.flatnonzero(y == label))
        n = max(1, int(len(idx) * fraction))
        if n >= len(idx):
            raise ValueError("Need at least two examples of each class")
        valid.extend(idx[:n])
        train.extend(idx[n:])
    return np.asarray(train), np.asarray(valid)


def fit_preprocessor(x):
    """Fit finite-value means/scales only on the training portion.

    This initial baseline treats numeric codes as numeric. Feature-specific BRFSS
    missing codes and categorical encodings need later, documented experiments.
    """
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    means = np.where(finite, x, 0).sum(axis=0) / np.maximum(counts, 1)
    clean = np.where(finite, x, means)
    scales = clean.std(axis=0)
    keep = (counts > 0) & (scales > 1e-12)
    return {"means": means, "scales": np.where(keep, scales, 1), "keep": keep}


def transform(x, state):
    clean = np.where(np.isfinite(x), x, state["means"])
    z = ((clean - state["means"]) / state["scales"])[:, state["keep"]]
    return np.column_stack([np.ones(len(z)), z])


def metrics(y, pred):
    tp = int(np.sum((y == 1) & (pred == 1)))
    fp = int(np.sum((y == -1) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == -1)))
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "accuracy": float(np.mean(y == pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def choose_threshold(y, scores):
    """Maximize positive-class validation F1 on a fixed 201-point score grid."""
    candidates = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    return float(
        max(candidates, key=lambda t: metrics(y, np.where(scores >= t, 1, -1))["f1"])
    )


def validate_submission(path, test_path):
    """Strict header, row count, label and exact test-ID order validation."""
    with open(path, newline="") as f, open(test_path, newline="") as t:
        reader, test = csv.reader(f), csv.reader(t)
        if next(reader) != ["Id", "Prediction"]:
            raise ValueError("Submission header must be Id,Prediction")
        next(test)
        seen = set()
        count = 0
        from itertools import zip_longest

        for prediction, example in zip_longest(reader, test):
            if prediction is None or example is None:
                raise ValueError("Submission and test row counts differ")
            if len(prediction) != 2 or prediction[1] not in {"-1", "1"}:
                raise ValueError("Each row needs an ID and a -1/+1 integer label")
            if prediction[0] != example[0] or prediction[0] in seen:
                raise ValueError("Mismatched, reordered, or duplicate test IDs")
            seen.add(prediction[0])
            count += 1
    if not count:
        raise ValueError("Empty submission")
    return count
