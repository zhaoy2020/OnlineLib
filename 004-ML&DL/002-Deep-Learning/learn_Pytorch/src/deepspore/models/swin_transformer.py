# This file is generated from saved notebook code blocks

#@swin_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinBlock1D(nn.Module):
    def __init__(self, dim, window_size, shift=False):
        super().__init__()
        self.window_size = window_size
        self.shift = shift

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape

        if self.shift:
            x = torch.roll(x, shifts=-self.window_size // 2, dims=1)

        # Pad to multiple of window size
        pad_len = (self.window_size - L % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_len))  # pad length dimension
        L_padded = x.shape[1]

        # Split into windows
        x_windows = x.view(B, L_padded // self.window_size, self.window_size, C)
        x_windows = x_windows.reshape(-1, self.window_size, C)  # [B*num_windows, window_size, C]

        # Attention
        shortcut = x_windows
        x_windows = self.norm1(x_windows)
        attn_out, _ = self.attn(x_windows, x_windows, x_windows)
        x_windows = shortcut + attn_out

        # Feed-forward
        shortcut = x_windows
        x_windows = self.norm2(x_windows)
        x_windows = shortcut + self.mlp(x_windows)

        # Merge windows
        x = x_windows.view(B, -1, C)
        if pad_len > 0:
            x = x[:, :L, :]

        if self.shift:
            x = torch.roll(x, shifts=self.window_size // 2, dims=1)
        return x


class SwinDNA(nn.Module):
    def __init__(self, input_dim=4, embed_dim=128, window_size=128, depth=4):
        super().__init__()
        self.embedding = nn.Conv1d(input_dim, embed_dim, kernel_size=15, padding=7)
        self.blocks = nn.ModuleList()

        for i in range(depth):
            shift = (i % 2 == 1)
            self.blocks.append(SwinBlock1D(embed_dim, window_size, shift=shift))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim, 2)  # Binary classification

    def forward(self, x):
        # x: [B, 4, L]
        x = self.embedding(x)  # [B, C, L]
        x = x.permute(0, 2, 1)  # [B, L, C]
        for block in self.blocks:
            x = block(x)
        x = x.permute(0, 2, 1)  # [B, C, L]
        x = self.pool(x).squeeze(-1)  # [B, C]
        return self.head(x)  # [B, 2]
