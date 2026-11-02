from loguru import logger 
import torch 
import torch.nn.functional as F 
from collections import Counter 


def collate_batch(batch, cut_off: int = 10_000_000):
    """Pads a list of (input_tensor, label) pairs to the same sequence length.
    Returns:
        - padded_inputs: (B, 4, L_max)
        - labels: (B,)
        - mask: (B, L_max), 1 for valid, 0 for padding
    """
    # (input1, input2, ..., input-batch_size)
    # (label1, label2, ..., label-batch_size)
    inputs, labels = zip(*batch)

    lengths = [x.shape[1] for x in inputs]  # each x is (4, L)
    max_len = max(lengths)
    # max_len = cut_off # Use cut_off to control the maximum length
    
    batch_size = len(inputs)
    padded = torch.zeros(size=(batch_size, 4, max_len), dtype=torch.float32)    # (B, 4, L_max)
    mask = torch.zeros(size=(batch_size, max_len), dtype=torch.bool)            # (B, L_max)
    for i, x in enumerate(inputs):
        padded[i, :, :x.shape[1]] = x   # left-align
        mask[i, : x.shape[1]] = 1       # valid tokens: 1, padding: 0
    
    # Control the cut off
    if padded.shape[2] > cut_off:
        padded = padded[:, :, :cut_off]
        mask = mask[:, :, :cut_off]

    return {'inputs':padded, 'labels':labels, 'attention_mask': mask} 


def get_sampler(dataset: torch.utils.data.Dataset, weights: list | None = None, labels: list | None = None):
    """Returns a sampler for the given dataset.
    Args:
        - dataset
        - weights
        - labels
    Return:
        - sampler
    """
    if weights is None:
        if labels is not None:
            label_counts = Counter(labels)
            weights = [1.0 / label_counts[label] for label in labels]
        else:
            raise ValueError("Labels is None.")
    sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    return sampler