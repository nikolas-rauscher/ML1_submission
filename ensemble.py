"""Save, load and combine the selected tree and neural network models."""

import json
from pathlib import Path

import numpy as np

from advanced_models import Binner, HistogramBoosting
from neural_model import MLP, mixed_encode


def save_bundle(path, bundle):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, payload=np.asarray(json.dumps(bundle, allow_nan=False)))


def load_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["payload"]))


def predict_component(component, x, names, batch_size=8192):
    if names != component["features"]:
        raise ValueError("Feature names or their order differ from training.")
    binner = Binner.from_dict(component["binner"])
    kind = component["model"]["type"]
    if kind == "histogram_boosting":
        model = HistogramBoosting.from_dict(component["model"])
        return model.predict(binner.transform(x, names))
    if kind == "numpy_mlp":
        model = MLP.from_dict(component["model"])
        return np.concatenate(
            [
                model.predict(
                    mixed_encode(
                        binner.transform(x[i : i + batch_size], names), binner.specs
                    )
                )
                for i in range(0, len(x), batch_size)
            ]
        )
    raise ValueError(f"Unknown model type: {kind}")


def predict_bundle(bundle, x, names):
    if bundle["schema_version"] != 1:
        raise ValueError("Unsupported model format.")
    weights = np.asarray(bundle["weights"], dtype=float)
    if (
        len(weights) != len(bundle["components"])
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1)
    ):
        raise ValueError("Invalid ensemble weights.")
    probabilities = np.zeros(len(x))
    for weight, component in zip(weights, bundle["components"]):
        probabilities += weight * predict_component(component, x, names)
    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain nonfinite values.")
    return probabilities
