from typing import Iterable, List, Dict, Tuple, Sequence
import random


def time_group_split(
    groups: Sequence[str],
    timestamps: Sequence[float],
    *,
    test_after: float,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """
    Create leak-safe splits with:
    - eTLD+1- (or group-) disjoint between splits
    - time-aware test strictly after a cutoff (test_after)

    Inputs:
    - groups: group ID per sample (e.g., eTLD+1). Samples with the same group never cross splits.
    - timestamps: a sortable numeric time per sample (e.g., UNIX seconds)
    - test_after: strictly greater timestamps go to test candidates
    - val_frac: fraction of pre-cutoff groups assigned to validation (rest to train)

    Returns dict with index lists: {"train": [...], "val": [...], "test": [...]}.
    """
    assert len(groups) == len(timestamps), "groups and timestamps must align"

    pre_idx = [i for i, t in enumerate(timestamps) if t <= test_after]
    post_idx = [i for i, t in enumerate(timestamps) if t > test_after]

    # Unique groups among pre-cutoff samples
    pre_groups = {}
    for i in pre_idx:
        g = groups[i]
        pre_groups.setdefault(g, []).append(i)

    rng = random.Random(seed)
    pre_group_keys = list(pre_groups.keys())
    rng.shuffle(pre_group_keys)

    val_count = int(round(val_frac * len(pre_group_keys)))
    val_groups = set(pre_group_keys[:val_count])
    train_groups = set(pre_group_keys[val_count:])

    train_idx: List[int] = []
    val_idx: List[int] = []

    for g, idxs in pre_groups.items():
        if g in val_groups:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)

    # Test: strictly post-cutoff, all groups allowed (do not mix with train/val due to time cut)
    test_idx = sorted(post_idx)

    return {"train": sorted(train_idx), "val": sorted(val_idx), "test": test_idx}


def export_split_indices(splits: Dict[str, List[int]]) -> Dict[str, List[int]]:
    """Thin wrapper to make the export explicit for serialization."""
    return {k: list(map(int, v)) for k, v in splits.items()}
