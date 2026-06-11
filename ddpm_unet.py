import torch
import torch.nn as nn
import math

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim)
        )
    def forward(self, t):                      # t: (B,) ints
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x, t_emb):
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]   # inject time
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)

class UNet(nn.Module):
    def __init__(self, base=64, time_dim=128):
        super().__init__()
        self.time_mlp = TimeEmbedding(time_dim)
        self.in_conv  = nn.Conv2d(1, base, 3, padding=1)         # (B,base,28,28)

        self.down1 = ResBlock(base,     base * 2, time_dim)      # then pool ->14
        self.down2 = ResBlock(base * 2, base * 4, time_dim)      # then pool ->7
        self.pool  = nn.AvgPool2d(2)

        self.mid   = ResBlock(base * 4, base * 4, time_dim)

        self.up    = nn.Upsample(scale_factor=2, mode="nearest")
        # decoder in_ch = upsampled + skip (concat), hence the sums
        self.up1   = ResBlock(base * 4 + base * 4, base * 2, time_dim)
        self.up2   = ResBlock(base * 2 + base * 2, base,     time_dim)

        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x  = self.in_conv(x)             # (B, base, 28,28)

        s1 = self.down1(x, t_emb)        # (B, 2base,28,28)  skip
        s2 = self.down2(self.pool(s1), t_emb)   # (B,4base,14,14) skip
        m  = self.mid(self.pool(s2), t_emb)     # (B,4base, 7, 7)

        h  = self.up(m)                              # 7 ->14
        h  = self.up1(torch.cat([h, s2], 1), t_emb)  # concat skip s2
        h  = self.up(h)                              # 14->28
        h  = self.up2(torch.cat([h, s1], 1), t_emb)  # concat skip s1
        return self.out_conv(h)          # (B,1,28,28) predicted noise
