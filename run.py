"""Create the exact CSV for AIcrowd submission 337745."""

import argparse
import json
from pathlib import Path

import numpy as np

from ensemble import load_bundle, predict_bundle
from helpers import create_csv_submission
from pipeline import load_features, sha256, validate_submission

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions.csv")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists. Choose a new --output path.")
    selected = json.loads((ROOT / "models/selected.json").read_text())
    model_path = ROOT / selected["path"]
    if sha256(model_path) != selected["sha256"]:
        parser.error("The saved model checksum does not match.")
    bundle = load_bundle(model_path)
    test_path = args.data_dir / "x_test.csv"
    if sha256(test_path) != bundle["test_sha256"]:
        parser.error("Test data differs from the official selected dataset.")
    ids, x, names = load_features(test_path)
    probabilities = predict_bundle(bundle, x, names)
    labels = np.where(probabilities >= bundle["threshold"], 1, -1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_csv_submission(ids, labels, args.output)
    rows = validate_submission(args.output, test_path)
    actual = sha256(args.output)
    if actual != bundle["prediction_sha256"]:
        raise RuntimeError(f"Prediction checksum differs: {actual}")
    print(f"Verified {rows} predictions: {args.output}\nSHA-256: {actual}")


if __name__ == "__main__":
    main()
