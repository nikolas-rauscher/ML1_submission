"""Train ridge regression and write predictions and validation results."""

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

from helpers import create_csv_submission
from implementations import ridge_regression
from pipeline import (
    choose_threshold,
    fit_preprocessor,
    load_features,
    load_labels,
    metrics,
    sha256,
    stratified_split,
    transform,
    validate_submission,
)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=433)
    parser.add_argument("--limit", type=int, help="Use only the first N training rows")
    args = parser.parse_args()
    if not args.name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.name
    ):
        parser.error(
            "--name must contain only letters, numbers, underscores or hyphens"
        )
    if args.lambda_ <= 0:
        parser.error("--lambda must be positive")
    if args.limit is not None and args.limit < 100:
        parser.error("--limit must be at least 100")
    out = ROOT / "runs" / args.name
    csv_path = out / "predictions.csv"
    if out.exists() or csv_path.exists():
        parser.error("Run name already exists. Choose a new --name.")
    for filename in ["x_train.csv", "y_train.csv", "x_test.csv"]:
        if not (args.data_dir / filename).is_file():
            parser.error(
                "Missing dataset. Put x_train.csv, y_train.csv and x_test.csv in --data-dir."
            )
    print("Loading training data...", flush=True)
    ids, x, feature_names = load_features(args.data_dir / "x_train.csv", args.limit)
    y = load_labels(args.data_dir / "y_train.csv", ids, args.limit)
    train, valid = stratified_split(y, args.seed)
    state = fit_preprocessor(x[train])
    print("Fitting validation baseline...", flush=True)
    w, _ = ridge_regression(y[train], transform(x[train], state), args.lambda_)
    scores = transform(x[valid], state) @ w
    threshold = choose_threshold(y[valid], scores)
    selected = metrics(y[valid], np.where(scores >= threshold, 1, -1))
    zero = metrics(y[valid], np.where(scores >= 0, 1, -1))
    majority = metrics(y[valid], -np.ones(len(valid)))
    print(
        json.dumps(
            {
                "validation_tuned_threshold": selected,
                "validation_zero_threshold": zero,
                "majority_baseline": majority,
                "threshold": threshold,
            },
            indent=2,
        ),
        flush=True,
    )
    print("Refitting on all available training rows...", flush=True)
    state = fit_preprocessor(x)
    w, loss = ridge_regression(y, transform(x, state), args.lambda_)
    del x
    test_ids, x_test, test_names = load_features(args.data_dir / "x_test.csv")
    if test_names != feature_names:
        raise ValueError("Train/test feature columns differ")
    pred = np.where(transform(x_test, state) @ w >= threshold, 1, -1)
    out.mkdir(parents=True)
    create_csv_submission(test_ids, pred, csv_path)
    rows = validate_submission(csv_path, args.data_dir / "x_test.csv")
    np.savez(out / "model.npz", w=w, threshold=threshold, **state)
    np.savez(out / "split.npz", train_ids=ids[train], validation_ids=ids[valid])
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit = None
    manifest = {
        "name": args.name,
        "model": "ridge_numeric_baseline",
        "lambda": args.lambda_,
        "seed": args.seed,
        "limit": args.limit,
        "training_rows": len(y),
        "test_rows": rows,
        "feature_count": len(feature_names),
        "threshold": threshold,
        "validation_selected": selected,
        "validation_zero_threshold": zero,
        "majority_baseline": majority,
        "validation_note": "The threshold was chosen on this validation set, so its F1 may be optimistic.",
        "training_mse": float(loss),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": commit,
        "submission_sha256": sha256(csv_path),
        "data_sha256": {
            f: sha256(args.data_dir / f)
            for f in ["x_train.csv", "y_train.csv", "x_test.csv"]
        },
        "code_sha256": {
            f: sha256(ROOT / f)
            for f in ["baseline.py", "pipeline.py", "implementations.py", "helpers.py"]
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {rows} predictions to {csv_path}.")


if __name__ == "__main__":
    main()
