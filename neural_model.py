"""NumPy neural network and survey-feature encoding."""

import time

import numpy as np

from advanced_models import best_f1, logloss, sigmoid


def mixed_encode(binned, specs):
    """One-hot categories; rank-scaled numeric bins plus missing indicator.

    All bin boundaries and category dictionaries were learned on fit rows.
    Missing numeric values are zero (middle rank), with an explicit indicator.
    """
    width = sum(len(s["cuts"]) + 1 if s["kind"] == "category" else 2 for s in specs)
    x = np.zeros((len(binned), width), dtype=np.float32)
    offset = 0
    rows = np.arange(len(binned))
    for j, spec in enumerate(specs):
        c = binned[:, j]
        missing = c == 255
        if spec["kind"] == "category":
            levels = len(spec["cuts"])
            x[rows, offset + np.where(missing, levels, c).astype(int)] = 1
            offset += levels + 1
        else:
            x[:, offset] = np.where(
                missing, 0, 2 * c.astype(np.float32) / max(len(spec["cuts"]), 1) - 1
            )
            x[:, offset + 1] = missing
            offset += 2
    return x


class MLP:
    """Two ReLU layers, a linear skip connection, dropout and Adam.

    BCE gradients are exact; L2 applies only to weight matrices, not biases.
    """

    def __init__(
        self,
        hidden=128,
        hidden2=64,
        dropout=0.1,
        lambda_=1e-4,
        learning_rate=0.001,
        batch_size=1024,
        epochs=100,
        patience=15,
        seed=433,
    ):
        self.config = dict(
            hidden=hidden,
            hidden2=hidden2,
            dropout=dropout,
            lambda_=lambda_,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            seed=seed,
        )

    def initialize(self, width):
        c = self.config
        rng = np.random.default_rng(c["seed"])
        self.params = {}
        for name, a, b in [
            ("w1", width, c["hidden"]),
            ("w2", c["hidden"], c["hidden2"]),
            ("w3", c["hidden2"], 1),
        ]:
            self.params[name] = (rng.normal(size=(a, b)) * np.sqrt(2 / a)).astype(
                np.float32
            )
        self.params.update(
            b1=np.zeros(c["hidden"], np.float32),
            b2=np.zeros(c["hidden2"], np.float32),
            b3=np.zeros(1, np.float32),
            skip=np.zeros((width, 1), np.float32),
        )

    def forward(self, x, rng=None):
        p, c = self.params, self.config
        z1 = x @ p["w1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        masks = []
        for layer in [1, 2]:
            a = a1 if layer == 1 else a2
            if rng is not None and c["dropout"]:
                mask = (rng.random(a.shape) < 1 - c["dropout"]).astype(np.float32) / (
                    1 - c["dropout"]
                )
                a *= mask
            else:
                mask = None
            masks.append(mask)
            if layer == 1:
                z2 = a1 @ p["w2"] + p["b2"]
                a2 = np.maximum(z2, 0)
        z = (a2 @ p["w3"] + p["b3"] + x @ p["skip"]).ravel()
        return z, (z1, a1, z2, a2, masks)

    def loss_grad(self, x, y01, rng=None):
        p, c = self.params, self.config
        z, (z1, a1, z2, a2, masks) = self.forward(x, rng)
        loss = np.mean(np.logaddexp(0, z) - y01 * z)
        dz = ((sigmoid(z) - y01) / len(x))[:, None]
        grads = {"w3": a2.T @ dz, "b3": dz.sum(axis=0), "skip": x.T @ dz}
        da2 = dz @ p["w3"].T
        if masks[1] is not None:
            da2 *= masks[1]
        da2 *= z2 > 0
        grads.update(w2=a1.T @ da2, b2=da2.sum(axis=0))
        da1 = da2 @ p["w2"].T
        if masks[0] is not None:
            da1 *= masks[0]
        da1 *= z1 > 0
        grads.update(w1=x.T @ da1, b1=da1.sum(axis=0))
        for key in ["w1", "w2", "w3", "skip"]:
            grads[key] += c["lambda_"] * p[key]
            loss += 0.5 * c["lambda_"] * np.sum(p[key] ** 2)
        return float(loss), grads

    def fit(self, x, y, x_val=None, y_val=None, callback=None):
        c = self.config
        self.initialize(x.shape[1])
        y01 = (y > 0).astype(np.float32)
        self.params["b3"][:] = np.log(y01.mean() / (1 - y01.mean()))
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        rng = np.random.default_rng(c["seed"])
        start = time.monotonic()
        step = 0
        best = -1.0
        best_loss = float("inf")
        best_epoch = 0
        best_params = None
        self.history = []
        self.fitted_epochs = 0
        for epoch in range(c["epochs"]):
            order = rng.permutation(len(y))
            # Halve the learning rate every 20 epochs.
            lr = c["learning_rate"] * (0.5 ** (epoch // 20))
            for begin in range(0, len(y), c["batch_size"]):
                rows = order[begin : begin + c["batch_size"]]
                loss, grads = self.loss_grad(x[rows], y01[rows], rng)
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite MLP loss")
                norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
                if not np.isfinite(norm):
                    raise FloatingPointError("Nonfinite MLP gradient")
                scale = min(1.0, 5.0 / max(norm, 1e-8))
                step += 1
                for key, g in grads.items():
                    g *= scale
                    m[key] *= 0.9
                    m[key] += 0.1 * g
                    v[key] *= 0.999
                    v[key] += 0.001 * g * g
                    self.params[key] -= (
                        lr
                        * (m[key] / (1 - 0.9**step))
                        / (np.sqrt(v[key] / (1 - 0.999**step)) + 1e-8)
                    )
            self.fitted_epochs = epoch + 1
            item = {
                "epoch": epoch + 1,
                "seconds": time.monotonic() - start,
                "learning_rate": lr,
            }
            if x_val is not None:
                pred = self.predict(x_val)
                score, _ = best_f1(y_val, pred)
                val_loss = logloss(y_val, pred)
                if score > best + 1e-5 or (
                    abs(score - best) <= 1e-5 and val_loss < best_loss
                ):
                    best, best_loss, best_epoch = score, val_loss, epoch + 1
                    best_params = {k: v.copy() for k, v in self.params.items()}
                item.update(
                    selection_f1=score,
                    selection_logloss=val_loss,
                    best_epoch=best_epoch,
                )
            self.history.append(item)
            if callback:
                callback(item)
            if x_val is not None and epoch + 1 - best_epoch >= c["patience"]:
                break
        if x_val is not None:
            self.params = best_params
            self.fitted_epochs = best_epoch
        self.elapsed = time.monotonic() - start
        return self

    def predict(self, x):
        return np.concatenate(
            [sigmoid(self.forward(x[i : i + 4096])[0]) for i in range(0, len(x), 4096)]
        )

    def to_dict(self):
        return {
            "type": "numpy_mlp",
            "config": self.config,
            "fitted_epochs": self.fitted_epochs,
            "params": {k: v.tolist() for k, v in self.params.items()},
        }

    @classmethod
    def from_dict(cls, state):
        obj = cls(**state["config"])
        obj.params = {
            k: np.asarray(v, dtype=np.float32) for k, v in state["params"].items()
        }
        obj.fitted_epochs = state["fitted_epochs"]
        return obj
