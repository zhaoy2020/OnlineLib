import math 
import torch 
from torch import nn 
import torch.nn.functional as F 
from loguru import logger 


class MaskLM(nn.Module):
    '''Masked Language Model.'''

    def __init__(self, vocab_size, num_inputs, num_hiddens):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        # 根据batch_idx和pred_positions从X中提取出对应的子序列
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        return self.mlp(masked_X)


class NextSentencePred(nn.Module):
    '''Whether the next sentence is a continuation of the previous one.'''

    def __init__(self, num_hiddens):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.Tanh()
        )
        self.output = nn.Linear(num_hiddens, 2)

    def forward(self, X):
        return self.output(self.hidden(X))
        

 
class BERTEncoder(nn.Module):
    '''Main BERT encoder model.'''

    def __init__(
            self, 
            vocab_size, 
            hidden_size, 
            num_layers, 
            num_heads, 
            dim_feedforward, 
        ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.segment_embedding = nn.Embedding(2, hidden_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, num_heads, dim_feedforward, batch_first=True),
            num_layers
        )

    def apply_rope(self, x):
        """RoPE for 1D sequence input: x of shape (B, L, D), D must be even."""
        B, L, D = x.shape
        assert D % 2 == 0, "Dimension must be even for RoPE"

        device = x.device
        half_d = D // 2

        pos = torch.arange(L, dtype=torch.float32, device=device)  # (L,)
        freq_seq = torch.exp(-torch.arange(0, half_d, 1.0, device=device) * math.log(10000.0) / half_d)  # (half_d,)
        angle = torch.einsum("l,d->ld", pos, freq_seq)  # (L, half_d)

        sin_pos = angle.sin().unsqueeze(0)  # (1, L, half_d)
        cos_pos = angle.cos().unsqueeze(0)  # (1, L, half_d)

        x1, x2 = x[..., :half_d], x[..., half_d:]  # (B, L, half_d)
        x_rotated = torch.cat([x1 * cos_pos - x2 * sin_pos, x2 * cos_pos + x1 * sin_pos], dim=-1)  # (B, L, D)
        return x_rotated

    def forward(self, tokens, segments, attention_mask=None):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.apply_rope(X)
        # attention_mask, 1 for padding.
        X = self.encoder(X, src_key_padding_mask=attention_mask)
        return X
    

class BERT(nn.Module):
    '''BERT model with MLM and Next Sentence Prediction heads.'''

    def __init__(
            self, 
            vocab_size, 
            hidden_size=768, 
            num_layers=2, 
            num_heads=12, 
            dim_feedforward=1028, 
        ):
        super().__init__()
        self.encoder = BERTEncoder(
            vocab_size, 
            hidden_size, 
            num_layers, 
            num_heads, 
            dim_feedforward, 
        )
        self.mlm_head = MaskLM(vocab_size, hidden_size, hidden_size)
        self.nsp_head = NextSentencePred(hidden_size)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        # BERTEncoder blocks
        encoded_X = self.encoder(tokens, segments, valid_lens)

        # MLM
        if pred_positions is not None:
            mlm_Y_hat = self.mlm_head(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None 

        # NSP
        nsp_Y_hat = self.nsp_head(encoded_X[:, 0, :])

        return encoded_X, mlm_Y_hat, nsp_Y_hat 