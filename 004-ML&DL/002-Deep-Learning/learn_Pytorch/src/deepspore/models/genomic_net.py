
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#@save
import torch 
from torch import nn 
import torch.nn.functional as F
import sys 
from loguru import logger 
from Pytorch.src.deepspore.matplotlib_config import set_plt_rcParams
from Pytorch.src.deepspore.training import parameter_sizes


#@save
def mask_conv1d(mask: torch.Tensor, kernel_size: int, stride: int, padding: int) -> torch.Tensor:
    '''Adjust the dimension with Conv1d for mask.'''
    mask = mask.float().unsqueeze(1)    # mask: (B, L) > (B, 1, L)
    mask = F.max_pool1d(mask, kernel_size=kernel_size, stride=stride, padding=padding)
    mask = mask.squeeze(1).bool()       # mask: (B, 1, L_out) > (B, L_out)
    return mask 


#@save
class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, num_channels: int, kernel_size: int = 3, stride: int = 1, use_1x1conv: bool = False, dropout: float = 0.1):
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride

        # CNN1
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=num_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)  # ↑ channels, ↓ size, 
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.activation1 = nn.GELU()
        self.pooling1 = nn.AvgPool1d(kernel_size=kernel_size, padding=kernel_size // 2, stride=1)
        self.dropout1 = nn.Dropout(dropout)

        # CNN2
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=kernel_size, padding=kernel_size // 2, stride=1) 
        self.bn2 = nn.BatchNorm1d(num_channels)
        self.activation2 = nn.GELU()
        self.pooling2 = nn.AvgPool1d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride) # ↓ size
        self.dropout2 = nn.Dropout(dropout)

        # CNN3
        self.conv3 = nn.Conv1d(in_channels, num_channels, kernel_size=1, stride=stride) if use_1x1conv else None # ↑ channels, ↓ size

    def forward(self, X, mask=None):
        """
        Args:
            X: shape (B, C, L)
            mask: shape (B, L), optional. True for valid, False for padding.
        Returns:
            Y: shape (B, C, L_out)
            mask_out: shape (B, L_out), if mask is given.
        """
        # CNN1
        Y = self.conv1(X)      # (B, C, L/stride)        
        if mask is not None:
            mask = mask_conv1d(mask, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride) # (B, L/stride)
        Y = self.bn1(Y)
        Y = self.activation1(Y)
        Y = self.pooling1(Y)
        Y = self.dropout1(Y)

        # CNN2
        Y = self.conv2(Y)      # (B, C, L/stride)
        if self.conv3:
            X = self.conv3(X)  # downsample residual path, X: (B, C, L/stride)
        Y += X                 # Residual Add
        Y = self.bn2(Y)
        Y = self.activation2(Y)  # (B, C, L/stride)
        Y = self.pooling2(Y)            # (B, C, L/(stride*stride))
        if mask is not None:
            mask = mask_conv1d(mask, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=self.stride) # mask: (B, L/(stride*stride))
        Y = self.dropout2(Y)

        # Y: (B, C, L(stride*stride))
        # mask: (B, L(stride*stride))
        # return (Y, mask) if mask is not None else Y
        return (Y, mask)


#@save
class Tower(nn.Module):
    '''Tower for genomic data.'''
    
    def __init__(self, in_channels: int, num_channels: int, num_towers: int):
        super().__init__()

        layers, self.final_channels = self.build_layers(in_channels, num_channels, num_towers)
        self.downsample = nn.ModuleList(layers)

    def build_layers(self, in_channels: int, num_channels: int, num_towers: int):
        layers = []
        layers.append(ResidualBlock(in_channels=in_channels, num_channels=num_channels, stride=1, use_1x1conv=True)) # First layer, keep the shape
        for i in range(1, num_towers):
            in_channels = num_channels
            num_channels *= 2
            # print(in_channels, num_channels)
            layers.append(ResidualBlock(in_channels=in_channels, num_channels=num_channels, stride=2, use_1x1conv=True))

        return layers, num_channels

    def forward(self, X, mask=None):
        for layer in self.downsample:
            X, mask = layer(X, mask)
        return X, mask




#@save
import math 


class FlashTransformerEncoder(nn.Module):
    '''FlashAttention Transformer Encoder Layer.'''

    def __init__(self, d_model: int=256, nhead: int=4, dim_feedforward: int=1024, dropout: float=0.1, num_layers: int=2):
        super().__init__()
        # FlashAttention encoder layer
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True  # FlashAttention requires norm_first=True
            ),
            num_layers=num_layers
        )

    def forward(self, X):
        X = self.encoder(X)
        return X


class StandardTransformerEncoder(nn.Module):
    '''Standard Transformer Encoder Layer.'''

    def __init__(self, d_model: int=256, nhead: int=4, dim_feedforward: int=1024, dropout: float=0.1, num_layers: int=2):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model= d_model, 
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            dropout= dropout,
            batch_first = True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer= block, 
            num_layers = num_layers,
        )

    def forward(self, X, src_key_padding_mask=None):
        X = self.encoder(X, src_key_padding_mask=src_key_padding_mask)
        return X
    

class Attn(nn.Module):
    '''Attention-based Transformer Encoder.'''
    
    def __init__(self, use_flash_attn: bool= False, d_model: int=256, nhead: int=4, dim_feedforward: int=1024, dropout: float=0.1, num_layers: int=2, use_position_rope: bool=False):
        super().__init__()

        # Using RoPE (Rotary Position Embedding) for positional encoding
        self.use_position_rope = use_position_rope
        if use_position_rope:
            assert d_model % 2 == 0, "d_model must be even for RoPE"
            logger.info("Using RoPE for positional encoding.")

        if use_flash_attn:
            raise NotImplementedError("FlashAttention is not implemented in this example.")
        else:
            self.encoder = StandardTransformerEncoder(
                d_model= d_model, 
                nhead= nhead,
                dim_feedforward= dim_feedforward,
                dropout= dropout,
                num_layers= num_layers,
            )

    def apply_rope(self, x):
        """
        RoPE for 1D sequence input: x of shape (B, L, D)
        D must be even.
        """
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

    def forward(self, X, src_key_padding_mask=None):
        # Position embedding
        if self.use_position_rope:
            X = self.apply_rope(X)

        # Transformer Encoder
        X = self.encoder(X, src_key_padding_mask=src_key_padding_mask)
        return X




#@save
class TaxonomyHeader(nn.Module):
    def __init__(self, embeded_dim: int, num_labels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embeded_dim, embeded_dim * 2),
            nn.GELU(),
            nn.Linear(embeded_dim * 2, embeded_dim),
            nn.GELU(),
            nn.Linear(embeded_dim, num_labels)
        )

    def forward(self, X):
        """
        Args:
            X: shape (B, D)
        Returns:
            logits: shape (B, num_labels)
        """
        logits = self.mlp(X)  # (B, num_labels)
        return logits




#@save
from torch.utils.checkpoint import checkpoint


class GenomicNet(nn.Module):
    def __init__(self, in_channels:int = 4, num_channels: int = 8, num_tower: int = 4, num_layers: int = 2, num_labels: int = 2, use_checkpoint: bool = False, prediction_token: str = 'average'):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        if use_checkpoint:
            logger.info("Using checkpoint.")

        # Input 
        self.input = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=num_channels, kernel_size=15, stride=2, padding=7), # ↑ channels, ↓ size
            nn.BatchNorm1d(num_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # Tower
        self.tower = Tower(in_channels=num_channels, num_channels=num_channels * 2, num_towers=num_tower)
        final_channels = self.tower.final_channels

        # Attention Pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, final_channels))  # learnable [CLS]
        self.attention = Attn(d_model=final_channels, nhead=4, dim_feedforward=1024, dropout=0.1, num_layers=num_layers, use_position_rope=True)  # 使用 Attention 模块

        self.prediction_token = prediction_token
        logger.info(f"Prediction token with {prediction_token}.")

        # Classifier
        self.classifier = TaxonomyHeader(embeded_dim=final_channels, num_labels=num_labels)

    def to_tokens(self, X):
        """
        Convert input sequence to tokens.
        X: (Batch, Channels, Length) -> (Batch, Length, Channels)
        """
        # X: (Batch, Channels, Length) -> (Batch, Length, Channels)
        X = X.permute(0, 2, 1)
        return X

    def add_cls(self, X, mask):
        '''
        Add CLS token to the input sequence.
        X: (Batch, Length, Channels) -> (Batch, Length+1, Channels)
        mask: (Batch, Length)
        '''
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(X.size(0), -1, -1)
        X = torch.cat((cls_tokens, X), dim=1)  # 在序列维度上拼接 CLS token, X: (Batch, Length+1, Channels)
        cls_tokens_mask = torch.tensor([1], device=X.device).expand(X.size(0), -1).bool()
        mask = torch.cat((cls_tokens_mask, mask), dim=1) # 在序列维度上拼接 CLS token mask (1 for valid), X: (Batch, Length+1)
        return X, mask
    
    def get_cls(self, X):
        '''
        Extract CLS token from the output sequence.
        X: (Batch, Length+1, Channels)
        Returns:
            cls_token: (Batch, Channels)
        '''
        # X: (B, cls+L, C)
        cls_token = X[:, 0, :]  # (B, C)
        return cls_token

    def average_pooling(self, Y, attention_mask=None):
        # Y: (B, L, C)
        if attention_mask is not None:
            mask_float = attention_mask.float().unsqueeze(-1)  # (B, L, 1)
            Y = (Y * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp(min=1e-6)  # 避免除0
        else:
            Y = Y.mean(dim=1) # (B, L, C) -> (B, C)
        return Y

    def forward(self, inputs, attention_mask=None, **kwargs):
        # X: (B, 4, L)
        # attention_mask: (B, L), 1 for valid and 0 for padding
        X = inputs
        logger.debug(f'shape of X: {X.shape}')

        if self.use_checkpoint:
            # X: (B, 4, L) -> Y: (B, 4, L/4)
            Y = checkpoint(self.input, X)
            attention_mask = mask_conv1d(mask=attention_mask, kernel_size=15, stride=2, padding=7)
            attention_mask = mask_conv1d(mask=attention_mask, kernel_size=3, stride=2, padding=1)
            logger.debug(f"Y/Mask shape after input: {Y.shape}, {attention_mask.shape}")

            # for layer in self.tower:
            #     Y, attention_mask = checkpoint(layer, (Y, attention_mask))
            Y, attention_mask = checkpoint(self.tower, Y, attention_mask)
            logger.debug(f"Y/Mask shape after tower: {Y.shape}, {attention_mask.shape}")

        else:
            # X: (B, 4, L) -> Y: (B, 4, L/4)
            Y = self.input(X)
            attention_mask = mask_conv1d(mask=attention_mask, kernel_size=15, stride=2, padding=7)
            attention_mask = mask_conv1d(mask=attention_mask, kernel_size=3, stride=2, padding=1)
            logger.debug(f"Y/Mask shape after input: {Y.shape}, {attention_mask.shape}")

            Y, attention_mask = self.tower(Y, attention_mask)
            logger.debug(f"Y/Mask shape after tower: {Y.shape}, {attention_mask.shape}")

        # To tokens for attention
        Y = self.to_tokens(Y) # (B, C, L) -> (B, L, C)

        if self.prediction_token == 'cls':
            # Add cls token and mask
            Y, attention_mask = self.add_cls(Y, mask=attention_mask)  # (B, L, C) -> (B, L+1, C)
            logger.debug(f'shape after adding CLS token: {Y.shape}')

            # Attention
            Y = self.attention(Y, src_key_padding_mask=~attention_mask) # (B, C, L) -> (B, L, C)
            logger.debug(f'shape after attention: {Y.shape}')
            
            # Take CLS token output
            Y = self.get_cls(Y)  # 取 CLS token 的输出, Y: (B, 1, C) -> (B, C)
            logger.debug(f'shape after CLS token: {Y.shape}')

        elif self.prediction_token == 'average':
            # Attention
            Y = self.attention(Y, src_key_padding_mask=~attention_mask) # (B, C, L) -> (B, L, C)
            logger.debug(f'shape after attention: {Y.shape}')

            # Average pooling over sequence length
            Y = self.average_pooling(Y, attention_mask)  # (B, L, C) -> (B, C)
            logger.debug(f'shape after pooling: {Y.shape}')

        # Classifier
        Y = self.classifier(Y) # Y: (B, C) -> (B, num_labels)
        logger.debug(f'shape after classifier: {Y.shape}')
        return Y 




#@save
from dataclasses import dataclass


@dataclass
class GenomicNetConfig:
    '''
    # 10_000_000 -> num_tower=5 
    # 7_000_000 -> num_tower=4
    # 1_000_000 -> num_tower=2
    '''
    in_channels: int = 4
    num_channels: int = 16
    num_tower: int = 4
    num_layers: int = 2
    num_labels: int = 2
    use_checkpoint: bool = False
    prediction_token: str = 'cls'

class GenomicNetConfigGA(GenomicNetConfig):
    num_labels: int = 4
    num_labels: int = 2

class GenomicNetConfigPM(GenomicNetConfig):
    num_labels: int = 4
    num_labels: int = 2

class GenomicNetConfigDA(GenomicNetConfig):
    num_labels: int = 4
    num_lables: int = 2

    
def get_model(config: GenomicNetConfig) -> GenomicNet:
    """    Get GenomicNet model with the given configuration.
    Args:
        config (GenomicNetConfig): Configuration for the GenomicNet model.
    Returns:
        GenomicNet: An instance of the GenomicNet model.
    """
    return GenomicNet(
        in_channels=config.in_channels,
        num_channels=config.num_channels,
        num_tower=config.num_tower,
        num_layers=config.num_layers,
        num_labels=config.num_labels,
        use_checkpoint=config.use_checkpoint,
        prediction_token=config.prediction_token
    )
