"""Histogram subtraction and best-first Newton tree growth.

Histograms store count, gradient and Hessian per feature/bin. Only the smaller
child is accumulated; its sibling is parent minus child. Split search is batched
over features. The mathematical objective matches advanced_models.split_search.
"""

import heapq
import time

import numpy as np

from advanced_models import HistogramBoosting, HistogramTree, goes_left


def histograms(x, g, h, rows, features):
    result = np.empty((3, len(features), 256), dtype=np.float64)
    gr, he = g[rows], h[rows]
    for k, j in enumerate(features):
        codes = x[rows, j]
        result[0, k] = np.bincount(codes, minlength=256)
        result[1, k] = np.bincount(codes, weights=gr, minlength=256)
        result[2, k] = np.bincount(codes, weights=he, minlength=256)
    return result


def best_split(hist, features, categorical, l2, min_leaf, min_hessian):
    """Vectorized numeric/missing and ordered-category split enumeration."""
    count, grad, hess = hist
    total_n = count.sum(axis=1)[:, None]
    total_g = grad.sum(axis=1)[:, None]
    total_h = hess.sum(axis=1)[:, None]
    parent = total_g**2 / (total_h + l2)

    def gains(gl, hl, nl):
        hr = np.maximum(total_h - hl, 0)
        gain = 0.5 * (gl**2 / (hl + l2) + (total_g - gl) ** 2 / (hr + l2) - parent)
        valid = (
            (nl >= min_leaf)
            & (total_n - nl >= min_leaf)
            & (hl >= min_hessian)
            & (hr >= min_hessian)
        )
        return np.where(valid, gain, -np.inf)

    gl = np.cumsum(grad[:, :255], axis=1)
    hl = np.cumsum(hess[:, :255], axis=1)
    nl = np.cumsum(count[:, :255], axis=1)
    right = gains(gl, hl, nl)
    left = gains(gl + grad[:, 255:], hl + hess[:, 255:], nl + count[:, 255:])
    cat = np.asarray(categorical, dtype=bool)[features]
    right[cat] = -np.inf
    left[cat] = -np.inf
    # Inactive bins are sorted last; all prefix positions after the last active
    # bin are invalid because the opposite child would contain zero rows.
    response = np.where(count > 0, grad / (hess + 10.0), np.inf)
    order = np.argsort(response, axis=1, kind="stable")
    cg = np.cumsum(np.take_along_axis(grad, order, axis=1), axis=1)[:, :-1]
    ch = np.cumsum(np.take_along_axis(hess, order, axis=1), axis=1)[:, :-1]
    cn = np.cumsum(np.take_along_axis(count, order, axis=1), axis=1)[:, :-1]
    category_gains = gains(cg, ch, cn)
    category_gains[~cat] = -np.inf
    all_gains = np.stack([right, left, category_gains], axis=1)
    k, kind, cut = np.unravel_index(np.argmax(all_gains), all_gains.shape)
    gain = all_gains[k, kind, cut]
    if not np.isfinite(gain) or gain <= 1e-10:
        return None
    return {
        "feature": int(features[k]),
        "gain": float(gain),
        "categories": order[k, : cut + 1].tolist() if kind == 2 else None,
        "threshold": -1 if kind == 2 else int(cut),
        "missing_left": bool(kind == 1),
    }


class FastHistogramTree(HistogramTree):
    def __init__(self, depth=6, min_leaf=100, l2=10.0, min_hessian=5.0, max_leaves=31):
        super().__init__(depth, min_leaf, l2, min_hessian)
        self.max_leaves = max_leaves if max_leaves is not None else 2**depth

    def fit(self, x, g, h, indices, features, categorical, deadline=float("inf")):
        self.nodes = []
        features = np.asarray(features, dtype=int)
        queue = []

        def check():
            if time.monotonic() >= deadline:
                raise TimeoutError("Tree fitting deadline")

        def add(rows, depth, hist):
            check()
            index = len(self.nodes)
            self.nodes.append(
                {
                    "feature": -1,
                    "value": float(
                        np.clip(g[rows].sum() / (h[rows].sum() + self.l2), -5, 5)
                    ),
                }
            )
            if depth < self.depth and len(rows) >= 2 * self.min_leaf:
                split = best_split(
                    hist,
                    features,
                    categorical,
                    self.l2,
                    self.min_leaf,
                    self.min_hessian,
                )
                if split is not None:
                    heapq.heappush(
                        queue, (-split["gain"], index, rows, depth, hist, split)
                    )
            return index

        check()
        add(indices, 0, histograms(x, g, h, indices, features))
        leaves = 1
        while queue and leaves < self.max_leaves:
            check()
            _, index, rows, depth, hist, split = heapq.heappop(queue)
            mask = goes_left(x[rows, split["feature"]], split)
            lr, rr = rows[mask], rows[~mask]
            if depth + 1 >= self.depth:
                lh = rh = None
            elif len(lr) <= len(rr):
                lh = histograms(x, g, h, lr, features)
                rh = hist - lh
            else:
                rh = histograms(x, g, h, rr, features)
                lh = hist - rh
            if lh is not None:
                # Subtraction can produce tiny negative Hessians at empty bins.
                lh[2] = np.maximum(lh[2], 0)
                rh[2] = np.maximum(rh[2], 0)
            left = add(lr, depth + 1, lh)
            right = add(rr, depth + 1, rh)
            self.nodes[index].update(split, left=left, right=right)
            leaves += 1
        return self


class FastHistogramBoosting(HistogramBoosting):
    def make_tree(self):
        c = self.config
        return FastHistogramTree(
            c["depth"], c["min_leaf"], c["l2"], max_leaves=c["max_leaves"]
        )
