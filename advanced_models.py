"""NumPy models for tabular data.

Logistic Newton boosting uses g=y-p, h=p*(1-p), leaf weight G/(H+l2),
and split gain (GL²/(HL+l2)+GR²/(HR+l2)-G²/(H+l2))/2.
"""

import time

import numpy as np

MISSING = 255
DROP_NAMES = {"SEQNO", "_PSU", "IDATE", "IYEAR"}


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def best_f1(y, scores):
    """Exact F1 threshold search, evaluating tied scores as an indivisible group."""
    y = np.asarray(y) > 0
    order = np.argsort(-scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    tp = np.cumsum(y[order])
    end = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    pos = np.flatnonzero(end)
    f1 = 2 * tp[pos] / np.maximum(pos + 1 + y.sum(), 1)
    k = int(np.argmax(f1))
    return float(f1[k]), float(sorted_scores[pos[k]])


def logloss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean((y > 0) * np.log(p) + (y <= 0) * np.log1p(-p)))


def clean_column(a, name):
    """Only documented, variable-specific recodes; never blanket 7/9 removal."""
    a = np.array(a, dtype=np.float64, copy=True)
    if name in {"PHYSHLTH", "MENTHLTH", "POORHLTH"}:
        a[a == 88] = 0
        a[np.isin(a, [77, 99])] = np.nan
    a[~np.isfinite(a)] = np.nan
    return a


class Binner:
    """Fit categories/quantile boundaries on training data only; explicit missing."""

    def __init__(self, max_bins=64, drop_rehab=False, category_limit=24):
        if not 2 <= max_bins <= 255 or not 2 <= category_limit <= 255:
            raise ValueError("Bins and category limit must fit uint8 with 255 reserved")
        self.max_bins = max_bins
        self.drop_rehab = drop_rehab
        self.category_limit = category_limit
        self.specs = []

    def fit(self, x, names):
        self.names = list(names)
        self.specs = []
        for j, name in enumerate(names):
            if name in DROP_NAMES or (self.drop_rehab and name == "HAREHAB1"):
                continue
            col = clean_column(x[:, j], name)
            finite = col[np.isfinite(col)]
            unique = np.unique(finite)
            if len(unique) < 2 and len(finite) == len(col):
                continue
            if not len(unique):
                continue
            if len(unique) <= self.category_limit:
                self.specs.append(
                    {
                        "index": j,
                        "name": name,
                        "kind": "category",
                        "cuts": unique.tolist(),
                    }
                )
            else:
                cuts = np.unique(
                    np.quantile(finite, np.linspace(0, 1, self.max_bins + 1)[1:-1])
                )
                self.specs.append(
                    {"index": j, "name": name, "kind": "numeric", "cuts": cuts.tolist()}
                )
        return self

    def transform(self, x, names=None):
        if names is not None and list(names) != self.names:
            raise ValueError("Feature schema differs from fitted model")
        b = np.full((len(x), len(self.specs)), MISSING, dtype=np.uint8, order="F")
        for j, spec in enumerate(self.specs):
            a = clean_column(x[:, spec["index"]], spec["name"])
            valid = np.isfinite(a)
            cuts = np.asarray(spec["cuts"])
            if spec["kind"] == "category":
                idx = np.searchsorted(cuts, a[valid])
                known = (idx < len(cuts)) & (
                    cuts[np.minimum(idx, len(cuts) - 1)] == a[valid]
                )
                rows = np.flatnonzero(valid)
                b[rows[known], j] = idx[known]
            else:
                b[valid, j] = np.searchsorted(cuts, a[valid], side="right")
        return b

    def to_dict(self):
        return {
            "max_bins": self.max_bins,
            "drop_rehab": self.drop_rehab,
            "category_limit": self.category_limit,
            "names": self.names,
            "specs": self.specs,
        }

    @classmethod
    def from_dict(cls, state):
        obj = cls(
            state["max_bins"], state["drop_rehab"], state.get("category_limit", 24)
        )
        obj.names, obj.specs = state["names"], state["specs"]
        return obj


def split_search(codes, g, h, indices, categorical, l2, min_leaf, min_hessian):
    """Find the best regularized split, including learned missing routing."""
    c = codes[indices]
    count = np.bincount(c, minlength=256)
    grad = np.bincount(c, weights=g[indices], minlength=256)
    hess = np.bincount(c, weights=h[indices], minlength=256)
    total_g, total_h, total_n = grad.sum(), hess.sum(), len(indices)
    parent = total_g**2 / (total_h + l2)
    if categorical:
        active = np.flatnonzero(count)
        if len(active) < 2:
            return None
        # Regularized Newton response ordering; includes missing as a category.
        order = active[np.argsort(grad[active] / (hess[active] + 10.0), kind="stable")]
        gl, hl, nl = (
            np.cumsum(grad[order])[:-1],
            np.cumsum(hess[order])[:-1],
            np.cumsum(count[order])[:-1],
        )
        gains = 0.5 * (
            gl**2 / (hl + l2) + (total_g - gl) ** 2 / (total_h - hl + l2) - parent
        )
        ok = (
            (nl >= min_leaf)
            & (total_n - nl >= min_leaf)
            & (hl >= min_hessian)
            & (total_h - hl >= min_hessian)
        )
        gains[~ok] = -np.inf
        k = int(np.argmax(gains))
        if not np.isfinite(gains[k]) or gains[k] <= 1e-10:
            return None
        return {
            "gain": float(gains[k]),
            "categories": order[: k + 1].tolist(),
            "threshold": -1,
            "missing_left": False,
        }
    # All nonmissing bins, with the missing mass assigned to either side.
    active = np.flatnonzero(count[:MISSING])
    if not len(active):
        return None
    end = int(active[-1]) + 1
    gl, hl, nl = np.cumsum(grad[:end]), np.cumsum(hess[:end]), np.cumsum(count[:end])
    best = None
    for missing_left in (False, True):
        a = gl + (grad[MISSING] if missing_left else 0)
        b = hl + (hess[MISSING] if missing_left else 0)
        n = nl + (count[MISSING] if missing_left else 0)
        gains = 0.5 * (
            a**2 / (b + l2) + (total_g - a) ** 2 / (total_h - b + l2) - parent
        )
        ok = (
            (n >= min_leaf)
            & (total_n - n >= min_leaf)
            & (b >= min_hessian)
            & (total_h - b >= min_hessian)
        )
        gains[~ok] = -np.inf
        k = int(np.argmax(gains))
        if (
            np.isfinite(gains[k])
            and gains[k] > 1e-10
            and (best is None or gains[k] > best["gain"])
        ):
            best = {
                "gain": float(gains[k]),
                "categories": None,
                "threshold": k,
                "missing_left": missing_left,
            }
    return best


def goes_left(codes, node):
    if node["categories"] is not None:
        lookup = np.zeros(256, dtype=bool)
        lookup[node["categories"]] = True
        return lookup[codes]
    return np.where(codes == MISSING, node["missing_left"], codes <= node["threshold"])


class HistogramTree:
    def __init__(self, depth=4, min_leaf=150, l2=20.0, min_hessian=5.0):
        self.depth, self.min_leaf, self.l2, self.min_hessian = (
            depth,
            min_leaf,
            l2,
            min_hessian,
        )
        self.nodes = []

    def fit(self, x, g, h, indices, features, categorical, deadline=float("inf")):
        self.nodes = []

        def grow(rows, depth):
            if time.monotonic() >= deadline:
                raise TimeoutError("Tree fitting deadline")
            i = len(self.nodes)
            self.nodes.append(
                {
                    "feature": -1,
                    "value": float(
                        np.clip(g[rows].sum() / (h[rows].sum() + self.l2), -5, 5)
                    ),
                }
            )
            if depth >= self.depth or len(rows) < 2 * self.min_leaf:
                return i
            best = None
            for j in features:
                split = split_search(
                    x[:, j],
                    g,
                    h,
                    rows,
                    categorical[j],
                    self.l2,
                    self.min_leaf,
                    self.min_hessian,
                )
                if split is not None and (best is None or split["gain"] > best["gain"]):
                    best = {**split, "feature": int(j)}
            if best is None:
                return i
            mask = goes_left(x[rows, best["feature"]], best)
            left, right = grow(rows[mask], depth + 1), grow(rows[~mask], depth + 1)
            self.nodes[i].update(best, left=left, right=right)
            return i

        grow(indices, 0)
        return self

    def predict(self, x):
        out = np.empty(len(x), dtype=np.float64)
        stack = [(0, np.arange(len(x)))]
        while stack:
            index, rows = stack.pop()
            node = self.nodes[index]
            if node["feature"] < 0:
                out[rows] = node["value"]
            else:
                mask = goes_left(x[rows, node["feature"]], node)
                stack.append((node["left"], rows[mask]))
                stack.append((node["right"], rows[~mask]))
        return out


def rank_features(binned, y, categorical):
    """Rank features by their Newton split gain on training rows."""
    p = np.mean(y > 0)
    g, h = (y > 0).astype(float) - p, np.full(len(y), p * (1 - p))
    rows = np.arange(len(y))
    scores = []
    for j in range(binned.shape[1]):
        split = split_search(binned[:, j], g, h, rows, categorical[j], 20, 100, 5)
        scores.append(split["gain"] if split else 0)
    return np.argsort(-np.asarray(scores), kind="stable"), np.asarray(scores)


class HistogramBoosting:
    def __init__(
        self,
        depth=4,
        learning_rate=0.08,
        min_leaf=150,
        l2=20,
        row_sample=0.85,
        col_sample=0.8,
        top_features=128,
        seed=433,
        n_estimators=500,
        patience=70,
        max_leaves=None,
        positive_weight=1.0,
    ):
        self.config = dict(
            depth=depth,
            learning_rate=learning_rate,
            min_leaf=min_leaf,
            l2=l2,
            row_sample=row_sample,
            col_sample=col_sample,
            top_features=top_features,
            seed=seed,
            n_estimators=n_estimators,
            patience=patience,
            max_leaves=max_leaves,
            positive_weight=positive_weight,
        )
        self.trees = []

    def make_tree(self):
        c = self.config
        return HistogramTree(c["depth"], c["min_leaf"], c["l2"])

    def fit(
        self,
        x,
        y,
        categorical,
        ranking,
        x_val=None,
        y_val=None,
        deadline=float("inf"),
        callback=None,
    ):
        c = self.config
        rng = np.random.default_rng(c["seed"])
        sample_weights = np.where(y > 0, c["positive_weight"], 1.0)
        prevalence = np.clip(np.average(y > 0, weights=sample_weights), 1e-6, 1 - 1e-6)
        self.base = float(np.log(prevalence / (1 - prevalence)))
        z = np.full(len(y), self.base)
        zv = None if x_val is None else np.full(len(x_val), self.base)
        feature_pool = ranking[: min(c["top_features"], x.shape[1])]
        best_score, best_loss, best_iteration = -1.0, float("inf"), 0
        start = time.monotonic()
        self.history, self.trees = [], []
        for step in range(c["n_estimators"]):
            if time.monotonic() >= deadline:
                break
            p = sigmoid(z)
            g, h = (y > 0).astype(float) - p, np.maximum(p * (1 - p), 1e-6)
            g *= sample_weights
            h *= sample_weights
            rows = (
                np.sort(
                    rng.choice(
                        len(y), max(2, int(len(y) * c["row_sample"])), replace=False
                    )
                )
                if c["row_sample"] < 1
                else np.arange(len(y))
            )
            features = np.sort(
                rng.choice(
                    feature_pool,
                    max(1, int(len(feature_pool) * c["col_sample"])),
                    replace=False,
                )
            )
            tree = self.make_tree()
            try:
                tree.fit(x, g, h, rows, features, categorical, deadline)
            except TimeoutError:
                break
            z += c["learning_rate"] * tree.predict(x)
            self.trees.append(tree)
            if x_val is not None:
                zv += c["learning_rate"] * tree.predict(x_val)
            elif callback and ((step + 1) % 10 == 0 or step == 0):
                callback({"iteration": step + 1, "seconds": time.monotonic() - start})
            if x_val is not None and ((step + 1) % 10 == 0 or step == 0):
                f1, threshold = best_f1(y_val, zv)
                loss = logloss(y_val, sigmoid(zv - np.log(c["positive_weight"])))
                if f1 > best_score + 1e-5 or (
                    abs(f1 - best_score) <= 1e-5 and loss < best_loss
                ):
                    best_score, best_loss, best_iteration = f1, loss, step + 1
                item = {
                    "iteration": step + 1,
                    "selection_f1": f1,
                    "selection_logloss": loss,
                    "best_iteration": best_iteration,
                    "seconds": time.monotonic() - start,
                }
                self.history.append(item)
                if callback:
                    callback(item)
                if step + 1 - best_iteration >= c["patience"]:
                    break
        if not self.trees:
            raise TimeoutError("No boosting tree was fitted")
        if x_val is not None:
            f1, _ = best_f1(y_val, zv)
            if f1 > best_score:
                best_iteration = len(self.trees)
            self.trees = self.trees[:best_iteration]
        self.elapsed = time.monotonic() - start
        return self

    def predict(self, x):
        z = np.full(len(x), self.base)
        for tree in self.trees:
            z += self.config["learning_rate"] * tree.predict(x)
        return sigmoid(z - np.log(self.config["positive_weight"]))

    def to_dict(self):
        return {
            "type": "histogram_boosting",
            "config": self.config,
            "base": self.base,
            "trees": [tree.nodes for tree in self.trees],
        }

    @classmethod
    def from_dict(cls, state):
        model = cls(**state["config"])
        model.base = state["base"]
        for nodes in state["trees"]:
            tree = HistogramTree()
            tree.nodes = nodes
            model.trees.append(tree)
        return model
