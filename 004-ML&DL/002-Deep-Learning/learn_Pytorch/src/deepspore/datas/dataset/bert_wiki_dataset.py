import torch 
from ..preprocessing.bert_wiki_mlm import _read_wiki, _get_nsp_data_from_paragraph, _get_mlm_data_from_tokens, _pad_bert_inputs                               
import d2l.torch as d2l


class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # 输入paragraphs[i]是代表段落的句子字符串列表；
        # 而输出paragraphs[i]是代表段落的句子列表，其中每个句子都是词元列表
        paragraphs = [d2l.tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        # 获取下一句子预测任务的数据
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(paragraph, paragraphs, self.vocab, max_len))
        # 获取遮蔽语言模型任务的数据
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)) for tokens, segments, is_next in examples]
        # 填充输入
        (self.all_token_ids, 
         self.all_segments, 
         self.valid_lens,
         self.all_pred_positions, 
         self.all_mlm_weights,
         self.all_mlm_labels, 
         self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], 
                self.all_segments[idx],
                self.valid_lens[idx], 
                self.all_pred_positions[idx],
                self.all_mlm_weights[idx], 
                self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)


def load_data_wiki(batch_size, data_dir, max_len):
    """加载WikiText-2数据集"""
    # data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    # data_dir = './data/wikipedia_text'
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=8)
    return train_iter, train_set.vocab, train_set 
