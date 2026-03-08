from __future__ import annotations

import numpy as np


def auc_roc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return None
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    rank_pos = ranks[: pos.size]
    u = rank_pos.sum() - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))
