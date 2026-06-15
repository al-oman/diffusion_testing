import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for shared modules
import torch
import torch.nn as nn
from ddpm.ddpm_unet import UNet


class CFGUNet(UNet):
    """Class-conditional UNet for classifier-free guidance.

    Reuses the unconditional UNet architecture wholesale (layers, ResBlock,
    TimeEmbedding) and only adds a label embedding that is summed into the
    time embedding. Index `num_classes` is the reserved null/unconditional
    token, so the same weights can produce both conditional and unconditional
    velocities depending on the label passed in.
    """
    def __init__(self, base=64, time_dim=128, num_classes=10):
        super().__init__(base=base, time_dim=time_dim)
        self.num_classes = num_classes
        self.null_class = num_classes                       # reserved index for the null label
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

    def forward(self, x, t, y):
        t_emb = self.time_mlp(t) + self.label_emb(y)         # inject class alongside time
        x  = self.in_conv(x)
        s1 = self.down1(x, t_emb)
        s2 = self.down2(self.pool(s1), t_emb)
        m  = self.mid(self.pool(s2), t_emb)
        h  = self.up(m)
        h  = self.up1(torch.cat([h, s2], 1), t_emb)
        h  = self.up(h)
        h  = self.up2(torch.cat([h, s1], 1), t_emb)
        return self.out_conv(h)
