"""Tests for feature encoding, trees and the neural network."""

import json
import time
import tempfile
import unittest
from pathlib import Path

import numpy as np

from advanced_models import (
    Binner,
    HistogramBoosting,
    HistogramTree,
    best_f1,
    rank_features,
    sigmoid,
    split_search,
)
from neural_model import MLP, mixed_encode
from train import prepare, fit_member
from fast_trees import FastHistogramBoosting, FastHistogramTree, best_split, histograms


class AdvancedTests(unittest.TestCase):
    def test_refit_artifact_and_inference(self):
        from ensemble import predict_component

        rng = np.random.default_rng(18)
        x = rng.normal(size=(200, 3))
        y = np.where(x[:, 0] * x[:, 1] > 0, 1, -1)
        names = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as folder:
            stage = Path(folder) / "fit"
            prepare(
                stage,
                x,
                y,
                names,
                dict(max_bins=16, category_limit=24),
                np.arange(len(y)),
            )
            recipe = dict(
                name="tree",
                kind="histogram_boosting",
                excluded_features=[],
                config=dict(depth=3, max_leaves=7, n_estimators=15, min_leaf=10, l2=1),
            )
            self.assertEqual(fit_member(str(stage), recipe), "tree")
            record = json.loads((stage / "tree.json").read_text())
            pred = predict_component(record["component"], x, names)
            self.assertTrue(np.isfinite(pred).all())
            self.assertGreater(np.mean((pred >= 0.5) == (y > 0)), 0.85)
            self.assertEqual(fit_member(str(stage), recipe), "tree")

    def test_mlp_analytic_gradient(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(13, 5))
        y = (rng.random(13) > 0.5).astype(float)
        model = MLP(hidden=4, hidden2=3, dropout=0, lambda_=0.01)
        model.initialize(5)
        model.params = {k: v.astype(float) for k, v in model.params.items()}
        _, grads = model.loss_grad(x, y)
        for key in model.params:
            for index in [0, model.params[key].size - 1]:
                original = model.params[key].flat[index]
                model.params[key].flat[index] = original + 1e-5
                plus, _ = model.loss_grad(x, y)
                model.params[key].flat[index] = original - 1e-5
                minus, _ = model.loss_grad(x, y)
                model.params[key].flat[index] = original
                self.assertAlmostEqual(
                    (plus - minus) / 2e-5, grads[key].flat[index], places=6
                )

    def test_mlp_learns_and_roundtrips(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(800, 5)).astype(np.float32)
        y = np.where(x[:, 0] * x[:, 1] > 0, 1, -1)
        m = MLP(
            hidden=24,
            hidden2=12,
            dropout=0,
            epochs=45,
            batch_size=100,
            learning_rate=0.01,
        ).fit(x, y)
        self.assertGreater(np.mean((m.predict(x) >= 0.5) == (y > 0)), 0.95)
        restored = MLP.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_array_equal(restored.predict(x), m.predict(x))

    def test_mixed_encoder_missing(self):
        specs = [
            dict(kind="category", cuts=[1, 2]),
            dict(kind="numeric", cuts=[1, 2, 3]),
        ]
        encoded = mixed_encode(
            np.array([[0, 0], [1, 3], [255, 255]], dtype=np.uint8), specs
        )
        np.testing.assert_array_equal(
            encoded,
            np.array([[1, 0, 0, -1, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]], np.float32),
        )

    def test_class_weight_prior_correction(self):
        x = np.zeros((100, 1), dtype=np.uint8)
        y = np.r_[np.ones(10), -np.ones(90)]
        m = FastHistogramBoosting(
            positive_weight=4, n_estimators=2, min_leaf=10, row_sample=1
        ).fit(x, y, [False], [0])
        np.testing.assert_allclose(m.predict(x), 0.1, atol=1e-12)

    def test_vectorized_splits_match_reference(self):
        rng = np.random.default_rng(812)
        for _ in range(12):
            x = rng.integers(0, 12, (601, 7), dtype=np.uint8)
            x[rng.random(x.shape) < 0.15] = 255
            g, h = rng.normal(size=len(x)), rng.uniform(0.01, 0.25, len(x))
            rows = np.arange(len(x))
            features = np.arange(x.shape[1])
            categorical = np.array([False, True, False, True, False, True, True])
            hist = histograms(x, g, h, rows, features)
            actual = best_split(hist, features, categorical, 3, 20, 1)
            reference = [
                split_search(x[:, j], g, h, rows, categorical[j], 3, 20, 1)
                for j in features
            ]
            gain = max(s["gain"] for s in reference if s is not None)
            self.assertAlmostEqual(actual["gain"], gain, places=10)
            small = histograms(x, g, h, rows[:100], features)
            np.testing.assert_allclose(
                hist - small, histograms(x, g, h, rows[100:], features), atol=1e-12
            )

    def test_tree_leaf_limit_and_model_save_load(self):
        rng = np.random.default_rng(91)
        x = rng.integers(0, 16, (1000, 4), dtype=np.uint8)
        y = np.where((x[:, 0] > 7) ^ (x[:, 1] > 7), 1, -1)
        m = FastHistogramBoosting(
            depth=5,
            max_leaves=7,
            min_leaf=10,
            l2=1,
            learning_rate=0.2,
            row_sample=1,
            col_sample=1,
            n_estimators=50,
        ).fit(x, y, [False] * 4, np.arange(4))
        self.assertGreater(np.mean((m.predict(x) >= 0.5) == (y > 0)), 0.97)
        for tree in m.trees:
            self.assertLessEqual(sum(n["feature"] < 0 for n in tree.nodes), 7)
        restored = HistogramBoosting.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_array_equal(m.predict(x), restored.predict(x))

    def test_exact_f1_handles_tied_scores(self):
        y = np.array([1, -1, 1, -1, -1])
        s = np.array([0.7, 0.7, 0.3, 0.2, 0.2])
        f1, cutoff = best_f1(y, s)
        brute = []
        for t in np.unique(s):
            p = s >= t
            tp = ((y == 1) & p).sum()
            brute.append(2 * tp / (p.sum() + (y == 1).sum()))
        self.assertAlmostEqual(f1, max(brute))
        self.assertEqual(cutoff, 0.3)

    def test_binner_does_not_learn_validation_categories(self):
        b = Binner().fit(
            np.array([[1, 88], [2, 3], [np.nan, 7.0]]), ["cat", "PHYSHLTH"]
        )
        before = json.dumps(b.to_dict())
        out = b.transform(np.array([[3, 77], [1, 88.0]]))
        self.assertEqual(out[0, 0], 255)
        self.assertEqual(out[0, 1], 255)
        self.assertNotEqual(out[1, 1], 255)
        self.assertEqual(before, json.dumps(b.to_dict()))

    def test_split_gain_matches_enumeration(self):
        codes = np.array([0, 0, 1, 1, 2, 2, 255, 255], dtype=np.uint8)
        g = np.array([-1.0, -1, 0, 0, 1, 1, 2, 2])
        h = np.ones(8)
        split = split_search(codes, g, h, np.arange(8), False, 2, 2, 1)
        brute = []
        for t in [0, 1, 2]:
            for missing_left in [False, True]:
                left = np.where(codes == 255, missing_left, codes <= t)
                if min(left.sum(), (~left).sum()) < 2:
                    continue
                gain = 0.5 * (
                    g[left].sum() ** 2 / (h[left].sum() + 2)
                    + g[~left].sum() ** 2 / (h[~left].sum() + 2)
                    - g.sum() ** 2 / (h.sum() + 2)
                )
                brute.append(gain)
        self.assertAlmostEqual(split["gain"], max(brute))

    def test_tree_learns_nonlinear_interaction_and_roundtrips(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(700, 3))
        y = np.where((x[:, 0] > 0) ^ (x[:, 1] > 0), 1, -1)
        b = Binner(16).fit(x, ["a", "b", "c"])
        xb = b.transform(x)
        categorical = [s["kind"] == "category" for s in b.specs]
        ranks, _ = rank_features(xb, y, categorical)
        m = HistogramBoosting(
            depth=3,
            min_leaf=10,
            l2=1,
            learning_rate=0.2,
            row_sample=1,
            col_sample=1,
            n_estimators=40,
        )
        m.fit(xb, y, categorical, ranks)
        p = m.predict(xb)
        self.assertGreater(np.mean((p >= 0.5) == (y == 1)), 0.9)
        restored = HistogramBoosting.from_dict(json.loads(json.dumps(m.to_dict())))
        np.testing.assert_array_equal(restored.predict(xb), p)

    def test_categorical_partition_and_missing(self):
        x = np.repeat(np.array([[0], [1], [2], [255]], dtype=np.uint8), 20, axis=0)
        g = np.repeat([-1.0, 1.0, -1.0, 1.0], 20)
        tree = HistogramTree(depth=1, min_leaf=10, l2=1).fit(
            x, g, np.ones(80), np.arange(80), [0], [True]
        )
        self.assertTrue(np.all(np.sign(tree.predict(x)) == np.sign(g)))

    def test_deadline_stops_before_tree(self):
        x = np.zeros((30, 1), dtype=np.uint8)
        with self.assertRaises(TimeoutError):
            HistogramBoosting().fit(
                x,
                np.r_[np.ones(15), -np.ones(15)],
                [False],
                [0],
                deadline=time.monotonic() - 1,
            )


if __name__ == "__main__":
    unittest.main()
