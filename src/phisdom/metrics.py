from typing import Sequence, Tuple, Optional


def _sorted_by_score(y_true: Sequence[int], y_score: Sequence[float]):
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    scores = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]
    return scores, labels


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute ROC-AUC via the probability-of-rank method (equivalent to Mann–Whitney U)."""
    assert len(y_true) == len(y_score) and len(y_true) > 0
    # Rank scores; handle ties by average rank
    scores, labels = _sorted_by_score(y_true, y_score)
    n = len(labels)
    # Assign ranks 1..n with averaging for ties
    ranks = [0.0] * n
    i = 0
    r = 1
    while i < n:
        j = i
        while j + 1 < n and scores[j + 1] == scores[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        r = j + 2
        i = j + 1
    pos = sum(1 for y in labels if y == 1)
    neg = n - pos
    if pos == 0 or neg == 0:
        return 0.5
    sum_ranks_pos = sum(rank for rank, y in zip(ranks, labels) if y == 1)
    u = sum_ranks_pos - pos * (pos + 1) / 2.0
    auc = u / (pos * neg)
    return float(auc)


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Compute area under the Precision-Recall curve (interpolated stepwise)."""
    assert len(y_true) == len(y_score) and len(y_true) > 0
    # Sort by descending score
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    tp = 0
    fp = 0
    fn = sum(1 for _, y in pairs if y == 1)
    if fn == 0:
        return 0.0
    prev_recall = 0.0
    prev_precision = 1.0
    auc = 0.0
    last_score = None
    for s, y in pairs:
        if last_score is not None and s != last_score:
            # Add area for segment between prev_recall and current recall (precision is piecewise constant)
            recall = tp / (tp + fn)
            precision = tp / max(1, tp + fp)
            auc += (recall - prev_recall) * prev_precision
            prev_recall = recall
            prev_precision = precision
        if y == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        last_score = s
    # Final segment
    recall = tp / (tp + fn if (tp + fn) > 0 else 1)
    precision = tp / max(1, tp + fp)
    auc += (recall - prev_recall) * prev_precision
    return float(max(0.0, min(1.0, auc)))


def fpr_at_tpr(y_true: Sequence[int], y_score: Sequence[float], target_tpr: float) -> Tuple[float, float]:
    """
    Return (fpr, threshold) at the smallest threshold achieving TPR >= target_tpr.
    If target_tpr is unattainable, returns (1.0, +inf).
    """
    assert 0.0 <= target_tpr <= 1.0
    pairs = sorted(zip(y_score, y_true), key=lambda x: -x[0])
    P = sum(1 for _, y in pairs if y == 1)
    N = len(pairs) - P
    if P == 0 or N == 0:
        return 1.0, float("inf")

    tp = 0
    fp = 0
    last_s = float("inf")
    for s, y in pairs + [(-float("inf"), None)]:
        if s != last_s:
            tpr = tp / P
            fpr = fp / N
            if tpr >= target_tpr:
                return fpr, last_s
            last_s = s
        if y == 1:
            tp += 1
        elif y == 0:
            fp += 1
    return 1.0, float("inf")


# ---- Safe wrappers for one-class splits ----

def _is_one_class(y_true: Sequence[int]) -> bool:
    try:
        s = set(int(y) for y in y_true)
    except Exception:
        s = set(y_true)
    return len(s) < 2


def roc_auc_safe(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Return ROC-AUC or None if split is one-class.

    Keeping the original roc_auc intact for callers that prefer sentinel values.
    """
    if _is_one_class(y_true):
        return float("nan")
    return roc_auc(y_true, y_score)


def pr_auc_safe(y_true: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """Return PR-AUC or None if split is one-class."""
    if _is_one_class(y_true):
        return float("nan")
    return pr_auc(y_true, y_score)
