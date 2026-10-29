"""Checks for leakage boundaries, metric conventions and malformed CSVs."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from implementations import reg_logistic_regression, ridge_regression
from pipeline import (
    fit_preprocessor,
    metrics,
    stratified_split,
    transform,
    validate_submission,
)


class PipelineTests(unittest.TestCase):
    def test_preprocessor_training_only_and_missing(self):
        x = np.array([[1, np.nan, 5], [3, np.nan, 5]], dtype=float)
        state = fit_preprocessor(x)
        np.testing.assert_allclose(transform(np.array([[5, 99, 5.0]]), state), [[1, 3]])
        np.testing.assert_allclose(
            transform(np.array([[np.nan, np.nan, 5.0]]), state), [[1, 0]]
        )
        np.testing.assert_allclose(state["means"], [2, 0, 5])

    def test_split_reproducible_disjoint_stratified(self):
        y = np.array([-1] * 80 + [1] * 20)
        a, b = stratified_split(y)
        c, d = stratified_split(y)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(len(a) + len(b), len(y))
        self.assertEqual(int((y[b] == 1).sum()), 4)

    def test_unpenalized_ridge_loss_and_normal_equation(self):
        x = np.array([[1.0, 2], [1, -1], [1, 0]])
        y = np.array([1.0, -1, 1])
        w, loss = ridge_regression(y, x, 0.2)
        np.testing.assert_allclose(x.T @ (x @ w - y) / len(y) + 0.4 * w, 0, atol=1e-12)
        self.assertAlmostEqual(float(loss), float(np.mean((y - x @ w) ** 2) / 2))

    def test_logistic_extreme_values_stable(self):
        x = np.array([[1.0], [-1.0]])
        w, loss = reg_logistic_regression(
            np.array([1.0, 0.0]), x, 0, np.array([1000.0]), 0, 0.1
        )
        self.assertTrue(np.isfinite(loss))

    def test_positive_class_f1(self):
        self.assertAlmostEqual(
            metrics(np.array([1, 1, -1, -1]), np.array([1, -1, 1, -1]))["f1"], 0.5
        )

    def test_submission_rejects_bad_labels_and_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            t, p = Path(folder) / "test.csv", Path(folder) / "pred.csv"
            t.write_text("Id,x\n10,0\n20,1\n")
            p.write_text("Id,Prediction\n10,-1\n20,1\n")
            self.assertEqual(validate_submission(p, t), 2)
            for bad in [
                "Id,Prediction\n10,0\n20,1\n",
                "Id,Prediction\n20,1\n10,-1\n",
                "Id,Prediction\n10,-1\n",
            ]:
                p.write_text(bad)
                with self.assertRaises(ValueError):
                    validate_submission(p, t)


if __name__ == "__main__":
    unittest.main()
