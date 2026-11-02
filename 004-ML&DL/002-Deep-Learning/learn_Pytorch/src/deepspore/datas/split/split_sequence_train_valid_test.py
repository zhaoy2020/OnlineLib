from sklearn.model_selection import train_test_split

def train_test_split_sequence(
    sequences: list,
    labels: list,
    valid_test_ratio: list[float] = [0.3, 0.15],
    seed: int = 42
):
    """
    Split sequences into train, valid, and test sets according to specified ratios,
    stratified by labels.

    Args:
        sequences (list): 数据样本
        labels (list): 样本对应的标签
        valid_test_ratio (list[float]): [valid_ratio, test_ratio]，其余为 train_ratio
        seed (int): 随机种子

    Returns:
        (train_seqs, train_labels), (valid_seqs, valid_labels), (test_seqs, test_labels)
    """
    train_ratio, valid_ratio, test_ratio = (
        1 - sum(valid_test_ratio),
        valid_test_ratio[0],
        valid_test_ratio[1]
    )

    assert train_ratio > 0 and valid_ratio >= 0 and test_ratio >= 0, "Invalid split ratios"
    assert abs(train_ratio + valid_ratio + test_ratio - 1) < 1e-6, "Split ratios must sum to 1"

    # Step 1: train + temp
    train_seqs, temp_seqs, train_labels, temp_labels = train_test_split(
        sequences, labels,
        test_size=(valid_ratio + test_ratio),
        stratify=labels,
        random_state=seed
    )

    # Step 2: valid + test
    valid_seqs, test_seqs, valid_labels, test_labels = train_test_split(
        temp_seqs, temp_labels,
        test_size=test_ratio / (valid_ratio + test_ratio),
        stratify=temp_labels,
        random_state=seed
    )

    return (train_seqs, train_labels), (valid_seqs, valid_labels), (test_seqs, test_labels)



if __name__ == '__main__':
    seqs = ["ATGC", "CGTA", "TTAA", "GGCC", "AATT", "CCGG"] * 100
    labels = [0, 1, 0, 1, 0, 1] * 100

    (train_seqs, train_labels), (valid_seqs, valid_labels), (test_seqs, test_labels) = \
        train_test_split_sequence(seqs, labels, valid_test_ratio=[0.2, 0.1])

    print("Train:", train_seqs, train_labels)
    print("Valid:", valid_seqs, valid_labels)
    print("Test:", test_seqs, test_labels)
