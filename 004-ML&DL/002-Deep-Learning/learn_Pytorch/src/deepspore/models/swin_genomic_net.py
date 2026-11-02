
    # Author: Yu Zhao
    # Copyright (c) 2025 Yu Zhao
    

#!SwinGenomicNet

import torch
import torch.nn as nn
import torch.nn.functional as F


#!SwinGenomicNet

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    
    def forward(self, x):
        # B, C, H, W -> B, embed_dim, H//p, W//p
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x



#!SwinGenomicNet

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=3):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B*num_windows, window_size*window_size, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, heads, N, N)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)



#!SwinGenomicNet

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0, mlp_ratio=4.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # B, H*W, C
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = self.window_partition(x)  # B*nW, ws*ws, C
        attn_windows = self.attn(x_windows)  # attention

        # merge windows
        x = self.window_reverse(attn_windows, H, W)  # B, H, W, C

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

    def window_partition(self, x):
        B, H, W, C = x.shape
        ws = self.window_size
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws * ws, C)
        return windows

    def window_reverse(self, windows, H, W):
        B = int(windows.shape[0] / (H * W / self.window_size / self.window_size))
        ws = self.window_size
        x = windows.view(B, H // ws, W // ws, ws, ws, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x



#!SwinGenomicNet

class SwinTransformerTiny(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim=96)
        self.blocks = nn.Sequential(
            SwinBlock(dim=96, input_resolution=(56, 56), num_heads=3),
            SwinBlock(dim=96, input_resolution=(56, 56), num_heads=3, shift_size=3),
        )
        self.norm = nn.LayerNorm(96)
        self.head = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(1)  # Global average pooling
        return self.head(x)



#!SwinGenomicNet