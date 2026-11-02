import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed1D(nn.Module):
    def __init__(self, patch_size=10, stride=None, in_chans=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride or patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=self.stride)
        
    def forward(self, x):
        x = self.proj(x)
        return x.transpose(1, 2)

class WindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1), num_heads)
        )
        coords = torch.arange(window_size)
        relative_coords = coords[:, None] - coords[None, :]
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(self, x):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + relative_position_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

class SwinBlock1D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        assert 0 <= shift_size < window_size, "shift_size must in [0, window_size)"
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention1D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
    
    def forward(self, x):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)
        
        # 确保能整除
        assert L % self.window_size == 0, f"Sequence length {L} must be divisible by window_size {self.window_size}"
        
        x_windows = x.view(B, L // self.window_size, self.window_size, C)
        x_windows = x_windows.view(-1, self.window_size, C)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(B, L // self.window_size, self.window_size, C)
        x = attn_windows.reshape(B, L, C)
        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)
        
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class PatchMerging1D(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(dim * 2, dim * 2)
        self.norm = nn.LayerNorm(dim * 2)
    
    def forward(self, x):
        B, L, C = x.shape
        assert L % 2 == 0, "input feature length must be even."
        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinTransformer1D(nn.Module):
    def __init__(
        self,
        input_length=1260,  # 修复：1260 = 10 * 126
        patch_size=10,
        in_chans=1,
        embed_dim=96,
        depths=[2, 6],      # 修复：2 个 stage（不是 3 个）
        num_heads=[3, 6],   # 修复：2 个 head 配置
        window_size=7,
        num_classes=4,
        dropout=0.1
    ):
        super().__init__()
        self.input_length = input_length
        self.patch_size = patch_size
        self.window_size = window_size
        self.num_layers = len(depths)
        
        # 验证尺寸兼容性
        assert input_length % patch_size == 0, "input_length must be divisible by patch_size"
        patch_num = input_length // patch_size
        
        # 关键修复：验证每个 stage
        for i in range(len(depths)):
            downsample_ratio = 2 ** i
            current_patches = patch_num // downsample_ratio
            assert current_patches % window_size == 0, \
                f"Stage {i}: patch_num={current_patches} must be divisible by window_size={window_size}"
        
        self.patch_embed = PatchEmbed1D(patch_size, patch_size, in_chans, embed_dim)
        patches_resolution = input_length // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, patches_resolution, embed_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = nn.ModuleList()
            dim = embed_dim * (2 ** i_layer)
            current_resolution = patches_resolution // (2 ** i_layer)
            for i_block in range(depths[i_layer]):
                shift_size = 0 if (i_block % 2 == 0) else window_size // 2
                layer.append(
                    SwinBlock1D(
                        dim=dim,
                        input_resolution=current_resolution,
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        shift_size=shift_size
                    )
                )
            if i_layer < self.num_layers - 1:
                layer.append(PatchMerging1D(current_resolution, dim))
            self.layers.append(nn.Sequential(*layer))
        
        final_dim = embed_dim * (2 ** (self.num_layers - 1))
        self.norm = nn.LayerNorm(final_dim)
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(final_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # 自动调整输入长度
        if x.size(-1) != self.input_length:
            if x.size(-1) > self.input_length:
                x = x[..., :self.input_length]
            else:
                x = F.pad(x, (0, self.input_length - x.size(-1)))
        
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for layer in self.layers:
            x = layer(x)
        
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.classifier(x)

# ==============================
# 测试修复后的模型
# ==============================
if __name__ == "__main__":
    # 1260 = 10 * 126
    # Stage 0: 126 % 7 = 0 ✅
    # Stage 1: 126 // 2 = 63, 63 % 7 = 0 ✅
    model = SwinTransformer1D(
        input_length=1260,    # 修复长度
        patch_size=10,
        embed_dim=96,
        depths=[2, 6],        # 修复：2 个 stage
        num_heads=[3, 6],     # 修复：2 个 head 配置
        window_size=7,
        num_classes=4
    )
    
    x = torch.randn(2, 1, 1340)  # 原始 1340 长度
    output = model(x)  # 自动裁剪到 1260
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # [2, 4]
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")