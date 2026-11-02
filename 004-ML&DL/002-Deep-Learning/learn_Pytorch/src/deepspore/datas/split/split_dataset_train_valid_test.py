'''
推荐实践:
- 数据量大 → 随机划分即可。
- 数据不平衡 → 用 分层划分。
- 有批次/病人/实验条件 → 必须 分组划分，否则会信息泄漏。
- 小数据集 → 可以 交叉验证 替代 valid/test。
'''


import torch
from torch.utils.data import Dataset, Subset, random_split
from sklearn.model_selection import train_test_split
from collections.abc import Sequence


def split_random(dataset: Dataset, train_ratio: float = 0.7, valid_ratio: float = 0.15, seed: int = 42) -> tuple[Subset]:
    """随机划分数据集 (不考虑类别分布)
    Example:
    >>> train_set, valid_set, test_set = split_random(dataset, train_ratio=0.7, valid_ratio=0.15)
    """
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_valid = int(valid_ratio * n_total)
    n_test = n_total - n_train - n_valid
    train_set, valid_set, test_set = random_split(dataset, [n_train, n_valid, n_test], generator=torch.Generator().manual_seed(seed))
    return train_set, valid_set, test_set


def split_stratified(dataset: Dataset, labels: Sequence, train_ratio: float = 0.7, valid_ratio: float = 0.15, seed: int = 42) -> tuple[Subset]:
    """分层划分 (保持类别比例一致)
    Example:
    >>> labels = [dataset[i][1] for i in range(len(dataset))]
    >>> train_set, valid_set, test_set = split_stratified(dataset, labels, train_ratio=0.7, valid_ratio=0.15)
    """
    indices = list(range(len(dataset)))

    # Step 1: train + temp
    train_idx, temp_idx = train_test_split(indices, test_size=1-train_ratio, stratify=labels, random_state=seed)

    # Step 2: valid + test
    temp_labels = [labels[i] for i in temp_idx]
    valid_ratio_adj = valid_ratio / (1-train_ratio)  # 在 temp 中再划分
    valid_idx, test_idx = train_test_split(temp_idx, test_size=1-valid_ratio_adj, stratify=temp_labels, random_state=seed)

    return Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)


def split_group(dataset: Dataset, groups: Sequence, train_ratio: float = 0.7, valid_ratio: float = 0.15, seed: int = 42) -> tuple[Subset]:
    """按分组划分 (如病人ID、实验批次，避免信息泄漏)
    Example:
    >>> groups = [metadata[i]["patient_id"] for i in range(len(dataset))]
    >>> train_set, valid_set, test_set = split_group(dataset, groups, train_ratio=0.7, valid_ratio=0.15)
    """
    import numpy as np

    unique_groups = np.unique(groups)

    # Step 1: train + temp
    train_groups, temp_groups = train_test_split(
        unique_groups, test_size=1-train_ratio, random_state=seed
    )

    # Step 2: valid + test
    valid_ratio_adj = valid_ratio / (1-train_ratio)
    valid_groups, test_groups = train_test_split(
        temp_groups, test_size=1-valid_ratio_adj, random_state=seed
    )

    train_idx = [i for i, g in enumerate(groups) if g in train_groups]
    valid_idx = [i for i, g in enumerate(groups) if g in valid_groups]
    test_idx  = [i for i, g in enumerate(groups) if g in test_groups]

    return Subset(dataset, train_idx), Subset(dataset, valid_idx), Subset(dataset, test_idx)
