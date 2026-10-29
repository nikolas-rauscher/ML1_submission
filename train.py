"""Fit the selected ensemble on all official training rows."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from advanced_models import Binner, rank_features
from ensemble import predict_bundle, save_bundle
from fast_trees import FastHistogramBoosting
from helpers import create_csv_submission
from neural_model import MLP, mixed_encode
from pipeline import load_features, load_labels, sha256, validate_submission

ROOT = Path(__file__).resolve().parent


def atomic_json(path, obj):
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_training():
    """Cache numeric CSV parsing, verifying raw hashes on every invocation."""
    cache = ROOT / ".cache/model-data"
    cache.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / "data/raw" / f for f in ("x_train.csv", "y_train.csv")]
    hashes = {p.name: sha256(p) for p in paths}
    meta = cache / "training.json"
    if meta.exists() and json.loads(meta.read_text())["sha256"] == hashes:
        info = json.loads(meta.read_text())
        return (
            np.load(cache / "ids.npy"),
            np.load(cache / "x.npy", mmap_mode="r"),
            np.load(cache / "y.npy"),
            info["features"],
            hashes,
        )
    ids, x, names = load_features(paths[0])
    y = load_labels(paths[1], ids)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate training IDs")
    np.save(cache / "ids.npy", ids)
    np.save(cache / "x.npy", x)
    np.save(cache / "y.npy", y)
    meta.write_text(json.dumps({"features": names, "sha256": hashes}, indent=2))
    return ids, np.load(cache / "x.npy", mmap_mode="r"), y, names, hashes


def fit_member(stage_path, recipe):
    stage = Path(stage_path)
    name = recipe["name"]
    dest = stage / (name + ".json")
    if dest.exists():
        saved = json.loads(dest.read_text())
        if saved["recipe"] != recipe:
            raise ValueError("Refit recipe changed")
        return name
    started = time.monotonic()

    def event(**fields):
        atomic_json(
            stage / (name + "_status.json"),
            dict(pid=os.getpid(), elapsed_seconds=time.monotonic() - started, **fields),
        )

    try:
        event(state="running", stage="loading")
        info = json.loads((stage / "prepared.json").read_text())
        y = np.load(stage / "y.npy")
        if recipe["kind"] == "histogram_boosting":
            b = np.load(stage / "binned.npy", mmap_mode="r")
            ranking = np.array(
                [
                    j
                    for j in info["ranking"]
                    if info["binner"]["specs"][j]["name"]
                    not in recipe["excluded_features"]
                ]
            )
            model = FastHistogramBoosting(**recipe["config"]).fit(
                b,
                y,
                info["categorical"],
                ranking,
                callback=lambda item: event(state="running", stage="fitting", **item),
            )
            if len(model.trees) != recipe["config"]["n_estimators"]:
                raise RuntimeError("Incomplete locked refit")
        elif recipe["kind"] == "numpy_mlp":
            x = np.load(stage / "mixed.npy", mmap_mode="r")
            model = MLP(**recipe["config"]).fit(
                x,
                y,
                callback=lambda item: event(state="running", stage="fitting", **item),
            )
            if model.fitted_epochs != recipe["config"]["epochs"]:
                raise RuntimeError("Incomplete locked refit")
        else:
            raise ValueError("Unsupported selected model family")
        component = dict(
            features=info["binner"]["names"],
            binner=info["binner"],
            model=model.to_dict(),
            excluded_features=recipe["excluded_features"],
        )
        atomic_json(
            dest, dict(recipe=recipe, component=component, fit_seconds=model.elapsed)
        )
        event(state="complete", stage="complete", fit_seconds=model.elapsed)
        return name
    except BaseException as exc:
        event(state="failed", stage="failed", error=repr(exc))
        raise


def prepare(stage, x, y, names, binner_config, ids):
    if (stage / "prepared.json").exists():
        return
    stage.mkdir(exist_ok=True)
    binner = Binner(**binner_config).fit(x, names)
    b = binner.transform(x)
    categorical = [s["kind"] == "category" for s in binner.specs]
    ranking, _ = rank_features(b, y, categorical)
    np.save(stage / "binned.npy", b)
    np.save(stage / "y.npy", y)
    np.save(stage / "ids.npy", ids)
    mixed = mixed_encode(b, binner.specs)
    np.save(stage / "mixed.npy", mixed)
    atomic_json(
        stage / "prepared.json",
        dict(
            binner=binner.to_dict(), ranking=ranking.tolist(), categorical=categorical
        ),
    )


def refit(stage, recipes, weights, workers, event):
    completed = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        jobs = [pool.submit(fit_member, str(stage), r) for r in recipes]
        for future in as_completed(jobs):
            completed.append(future.result())
            event(stage.name + "_refitting", completed_members=completed)
    components = [
        json.loads((stage / (r["name"] + ".json")).read_text())["component"]
        for r in recipes
    ]
    return dict(schema_version=1, components=components, weights=weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name", default="train_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not args.name or any(
        c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for c in args.name
    ):
        parser.error("Use only letters, numbers, underscores or hyphens in the name.")
    if not 1 <= args.workers <= 6:
        parser.error("Choose 1 to 6 workers.")
    plan = json.loads((ROOT / "models/recipe.json").read_text())
    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=False)

    def event(stage, **fields):
        status = dict(stage=stage, pid=os.getpid(), **fields)
        atomic_json(out / "status.json", status)
        print(json.dumps(status), flush=True)

    event("loading")
    ids, x, y, names, hashes = load_training()
    if hashes != plan["data_sha256"]:
        raise ValueError("Training data differs from the selected dataset.")
    event("preparing")
    prepare(out / "fit", x, y, names, plan["binner_config"], ids)
    bundle = refit(out / "fit", plan["recipes"], plan["weights"], args.workers, event)
    test_path = ROOT / "data/raw/x_test.csv"
    if sha256(test_path) != plan["test_sha256"]:
        raise ValueError("Test data differs from the selected dataset.")
    test_ids, test_x, test_names = load_features(test_path)
    probabilities = predict_bundle(bundle, test_x, test_names)
    output = out / "predictions.csv"
    create_csv_submission(
        test_ids, np.where(probabilities >= plan["threshold"], 1, -1), output
    )
    rows = validate_submission(output, test_path)
    if sha256(output) != plan["prediction_sha256"]:
        raise ValueError("Retrained predictions differ from the selected CSV.")
    bundle.update(
        threshold=plan["threshold"],
        test_sha256=plan["test_sha256"],
        prediction_sha256=plan["prediction_sha256"],
        training_rows=len(y),
    )
    save_bundle(out / "model.npz", bundle)
    event("complete", rows=rows, csv=str(output))


if __name__ == "__main__":
    main()
