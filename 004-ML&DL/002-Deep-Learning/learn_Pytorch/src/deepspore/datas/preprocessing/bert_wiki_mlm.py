import os 
import pandas as pd 
import random
import torch


def _read_wiki(data_dir):
    # file_name = os.path.join(data_dir, 'wiki.train.tokens')
    # with open(file_name, 'r') as f:
    #     lines = f.readlines()
    file_name = os.path.join(data_dir, 'train-00000-of-00001.parquet')
    df = pd.read_parquet(data_dir)
    lines = df['text'].tolist()
    # 大写字母转换为小写字母
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs.
    Args:
        tokens_a: List[str] 第一段文本的词元列表（即第一句话的分词结果）。
        tokens_b: List[str] （可选）第二段文本的词元列表（即第二句话的分词结果）。在单句任务中，此参数可以省略。
    Returns:
        tokens: List[str] 词元列表，其中第一个词元是'<cls>'，表示序列的开始，最后一个词元是'<sep>'，表示序列的结束。
        segments: List[int] 段标识列表，其中0表示第一段，1表示第二段。
    """
    # classification (cls) and separator (sep) tokens are added
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _get_next_sentence(sentence, next_sentence, paragraphs):
    '''生成下一句预测任务的数据
    Args:
        sentence: 当前句子
        next_sentence: 下一句句子
        paragraphs: 段落列表
    Returns:
        sentence: 当前句子
        next_sentence: 下一句句子
        is_next: 是否是下一句
    '''
    # 1. 随机决定是否使用真实的下一句：使用 random.random() 生成一个 [0,1) 之间的随机数，如果随机数小于 0.5，则 is_next 设置为 True，表示使用真实的下一句
    if random.random() < 0.5:
        is_next = True
    else:
        # 生成负样本（不真实的下一句）：从 paragraphs 中随机选择一个段落，再从中随机选择一句作为 next_sentence。
        ## 设置 is_next 为 False，表示 next_sentence 不是 sentence 的真实下一句。
        # paragraphs是三重列表的嵌套
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    '''生成下一句预测任务的数据
    Args:
        paragraph: 段落, 当前处理的段落，通常是由多个句子组成的列表。
        paragraphs: 段落列表, 所有段落的集合，用于在生成负样本时随机选择其他句子。
        vocab: 词汇表, 用于将词元转换为对应的索引或其他形式。
        max_len: 最大长度, 模型接受的最大句子对长度，包含了特殊词元 <cls> 和 <sep>。
    Returns:
        nsp_data_from_paragraph: 下一句预测任务的数据
    '''
    nsp_data_from_paragraph = []
    # 遍历当前段落中的每一个句子，除了最后一个句子，因为需要成对处理句子与下一句的关系。
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(sentence=paragraph[i], next_sentence=paragraph[i + 1], paragraphs=paragraphs)
        # 考虑1个'<cls>'词元和2个'<sep>'词元
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    '''为遮蔽语言模型的输入创建新的词元副本，其中输入可能包含替换的“<mask>”或随机词元
    Args:
        tokens :list: 输入词元序列列表，通常是一个句子的词元化结果。
        candidate_pred_positions :list: 候选预测位置列表，表示哪些词元有可能被遮蔽和预测。
        num_mlm_preds :int: 需要遮蔽和预测的词元数量。
        vocab :Vocab: 词汇表对象，包含词元到索引的映射（idx_to_token）。
    Returns:
        mlm_input_tokens: 返回修改后的词元序列，包含被替换的词元。
        pred_positions_and_labels: 返回被替换词元的位置及其对应的原始词元，用于模型训练时的预测目标。
    '''
    # 创建一个输入词元的副本，准备在其中进行替换操作。
    mlm_input_tokens = [token for token in tokens]
    # 初始化一个空列表，用于存储被遮蔽词元的位置及其原始标签。
    pred_positions_and_labels = []
    # 打乱后用于在遮蔽语言模型任务中获取15%的随机词元进行预测
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        # 如果已经替换了所需数量的词元（num_mlm_preds），则退出循环。
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80%的时间：将词替换为“<mask>”词元
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                # 10%的时间：保持词不变
                masked_token = tokens[mlm_pred_position]
            else:
                # 10%的时间：用随机词替换该词
                masked_token = random.choice(vocab.idx_to_token)
        # 替换操作: 根据上述概率策略，对指定位置的词元进行替换。
        mlm_input_tokens[mlm_pred_position] = masked_token
        # 将预测位置 及其 原始标签添 加到列表中，用于后续的损失计算和模型训练。
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    '''生成遮蔽语言模型任务的数据
    Args:
        tokens :list: 输入词元序列列表，通常是一个句子的词元化结果。
        vocab :Vocab: 词汇表对象，包含词元到索引的映射（idx_to_token）。
    Returns:
        vocab[mlm_input_tokens]：经过遮蔽处理后的词元序列，通常会被转换为词汇表中的索引。
        pred_positions：被遮蔽词元的位置索引列表。
        vocab[mlm_pred_labels]：被遮蔽词元的原始标签，通常也是词汇表中的索引。
    '''
    # 初始化候选预测位置列表
    candidate_pred_positions = []
    # 过滤特殊词元
    ## tokens是一个字符串列表
    for i, token in enumerate(tokens):
        # 在遮蔽语言模型任务中不会预测特殊词元
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 遮蔽语言模型任务中预测15%的随机词元
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    # 生成遮蔽后的词元和标签
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab)
    # 对被遮蔽的位置和标签进行排序,按照位置索引进行排序，确保顺序的一致性。这对于后续处理和训练时的批量操作非常重要。
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    # 分离位置索引和标签
    ## pred_positions：仅包含被遮蔽词元的位置索引。
    pred_positions = [v[0] for v in pred_positions_and_labels]
    ## mlm_pred_labels：包含这些位置上被遮蔽词元的原始标签。
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def _pad_bert_inputs(examples, max_len, vocab):
    '''为BERT模型的预训练任务（包括遮蔽语言模型任务 MLM 和下一句预测任务 NSP）准备和填充输入数据
    Args:
        examples :list:  包含多个样本的数据集。每个样本是一个元组，通常包含以下内容：
                        - token_ids：词元ID列表，表示一个句子的词元序列。
                        - pred_positions：被遮蔽词元的位置索引列表。
                        - mlm_pred_label_ids：被遮蔽词元的原始标签列表。
                        - segments：段落ID列表，表示句子在段落中的位置。
                        - is_next：布尔值，表示是否为下一句预测任务的标签。
        max_len :int: 模型接受的最大句子对长度，包含了特殊词元 <cls> 和 <sep>,所有样本将被填充或截断到这个长度。
        vocab :Vocab: 词汇表对象，包含词元到索引的映射（idx_to_token）。
    Returns:
        all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels
    '''
    # 根据最大长度 max_len 计算出最多可以进行遮蔽预测的词元数量，通常占总长度的15%。
    max_num_mlm_preds = round(max_len * 0.15)
    # 初始化列表，用于存储填充后的数据
    ## all_token_ids: 存储填充后的词元索引。
    ## all_segments: 存储填充后的段落ID。
    ## valid_lens: 存储每个样本的有效长度，不包括'<pad>'的计数。    
    all_token_ids, all_segments, valid_lens = [], [], []
    ## all_pred_positions: 存储填充后的被遮蔽词元的位置索引。
    ## all_mlm_weights: 存储填充后的被遮蔽词元的权重。
    ## all_mlm_labels: 存储填充后的被遮蔽词元的原始标签。
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    ## nsp_labels: 存储填充后的下一句预测任务的标签。
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        # 词元索引填充: 若 token_ids 的长度小于 max_len，则使用 <pad> 词元的索引进行填充，确保每个序列长度一致。
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        # 段落ID填充: 若 segments 的长度小于 max_len，则使用0进行填充，确保每个序列长度一致。
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        # valid_lens: 记录每个样本中实际有效的词元数量，即不包括填充的 <pad> 词元的数量。
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        # 被遮蔽词元的位置索引填充: 若 pred_positions 的长度小于 max_num_mlm_preds，则使用0进行填充，确保每个序列长度一致。
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # MLM 权重填充: 对于实际的遮蔽词元位置，赋予权重 1.0；对于填充的位置，赋予权重 0.0。这样在计算损失时，填充部分不会影响结果。
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        # MLM 标签填充: 对于实际的遮蔽词元，使用其真实的词元索引作为标签；对于填充的位置，使用 0 作为占位符。
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        # NSP 标签: 记录每个样本的下一句预测标签，通常为 0 或 1，表示是否为真实的下一句。
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)