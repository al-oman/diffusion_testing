import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for shared modules
import glob, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join
from utils import MnistDataloader
from ddpm.ddpm_unet import UNet

device = torch.device("mps")
TIME_SCALE = 1000.0   # must match flow_matching.py

input_path = './input'
(_, _), (x_test, _) = MnistDataloader(
    join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
    join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
    join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
    join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')).load_data()
x_test = torch.from_numpy(x_test[:1024]).float().to(device)[:, None] / 127.5 - 1

ckpts = sorted(glob.glob("checkpoints/fm_v2/fm_*.pth"),
               key=lambda p: int(re.search(r"fm_(\d+)", p).group(1)))
unet = UNet().to(device)

n, steps = 16, 100
dt = 1.0 / steps

# u_t loss evaluation of checkpoints
torch.manual_seed(0)  # same (t, x_0) draws for every checkpoint so losses are comparable
x_0 = torch.randn_like(x_test)
ts = torch.rand(x_test.shape[0], device=device)
t = ts.view(-1, 1, 1, 1)
x_t = (1-t)*x_0 + t*x_test
us = x_test - x_0
with torch.no_grad():
    for ckpt in ckpts:
        unet.load_state_dict(torch.load(ckpt, map_location=device)["ema"])
        unet.eval()
        loss = torch.mean((unet(x_t, ts * TIME_SCALE) - us)**2).item()
        print(f"{ckpt}: test u_t loss = {loss:.4f}")

unet.load_state_dict(torch.load(ckpts[-1], map_location=device)["ema"])
unet.eval()


x = torch.randn(n, 1, 28, 28, device=device)

# generating
with torch.no_grad():
    for i in range(steps):
        t = i * dt
        t_batch = torch.full((n,), t, device=device)
        v1 = unet(x, t_batch * TIME_SCALE)
        v2 = unet(x + v1*dt, (t_batch + dt) * TIME_SCALE)
        x = x + (v1 + v2)/2 * dt

# plotting
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, im in zip(axes.flat, x.clamp(-1, 1).cpu().numpy()[:, 0]):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
fig.suptitle(f"samples from {ckpts[-1]}")
epoch = re.search(r"fm_(\d+)", ckpts[-1]).group(1)
out_path = f"output/fm/fm_steps{steps}_ep{epoch}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")
plt.show()