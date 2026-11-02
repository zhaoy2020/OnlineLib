# This file is generated from saved notebook code blocks

#@augumentation
def reverse_complement(one_hot_seq):
    # one_hot_seq: shape (4, L), A=0, C=1, G=2, T=3
    # A <-> T, C <-> G
    idx_map = [3, 2, 1, 0]  # ACGT -> TGCA
    reversed_seq = torch.flip(one_hot_seq, dims=[1])
    return reversed_seq[idx_map]



#@augumentation
def circular_crop(seq_tensor, crop_len):
    ''' Crop a circular sequence tensor to a fixed length.
    Args:
        seq_tensor (torch.Tensor): Input sequence tensor of shape (C, L).
        crop_len (int): Length of the cropped sequence.
    Returns:
        torch.Tensor: Cropped sequence tensor of shape (C, crop_len).
    '''
    # seq_tensor: shape (C, L)
    double_seq = torch.cat([seq_tensor, seq_tensor], dim=1)
    L = seq_tensor.shape[1]
    start = torch.randint(0, L, (1,)).item()
    return double_seq[:, start:start + crop_len]
