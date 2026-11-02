import torch 


def to_mask(valid_lens_x: torch.Tensor, max_len: int):
    # valid_lens_x: batch_size
    batch_size = valid_lens_x.shape[0]
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    for i in range(batch_size):
        mask[i, :valid_lens_x[i].int()] = 1
    return mask


def _get_batch_loss_bert(net, loss, vocab_size, tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y, nsp_y):
    ''' 
    计算BERT模型的遮蔽语言模型损失和下一句预测任务损失。
    Args:
        net :BERTModel: BERT模型实例。
        loss :nn.CrossEntropyLoss: 损失函数实例。
        vocab_size :int: 词汇表大小。
        tokens_X :torch.Tensor: 输入词元索引。
        segments_X :torch.Tensor: 输入段落ID。
        valid_lens_x :torch.Tensor: 有效长度。
    Returns:
        mlm_l :torch.Tensor: 遮蔽语言模型损失。
        nsp_l :torch.Tensor: 下一句预测任务损失。
        l :torch.Tensor: 总损失。
    '''
    # 前向传播
    max_len = tokens_X.shape[1]
    attention_mask = to_mask(valid_lens_x=valid_lens_x, max_len=max_len)
    # _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X)
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X, ~attention_mask, pred_positions_X)
    
    # 计算遮蔽语言模型损失
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) * mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    
    # 计算下一句子预测任务的损失
    nsp_l = loss(nsp_Y_hat, nsp_y)
    
    l = mlm_l + nsp_l
    
    return mlm_l, nsp_l, l