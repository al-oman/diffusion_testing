import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for shared modules
import glob, re, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from cfg_unet import CFGUNet

parser = argparse.ArgumentParser(description="Classifier-free guided sampling from a conditional flow-matching model")
parser.add_argument("--target_number", type=int, default=8, help="digit class to generate")
parser.add_argument("--guidance_scale", type=float, default=3.0, help="CFG strength w (0=uncond, 1=plain cond, >1=amplified)")
args = parser.parse_args()
target_number = args.target_number
w = args.guidance_scale

device = torch.device("mps")
TIME_SCALE = 1000.0   # must match fm_cfg.py
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES

# Load conditional U-Net
ckpts = sorted(glob.glob("checkpoints/fm_cfg/fm_cfg_*.pth"),
               key=lambda p: int(re.search(r"fm_cfg_(\d+)", p).group(1)))
unet = CFGUNet(num_classes=NUM_CLASSES).to(device)
unet.load_state_dict(torch.load(ckpts[-1], map_location=device)["ema"])
unet.eval()

n, steps = 16, 100
dt = 1.0 / steps

x = torch.randn(n, 1, 28, 28, device=device)
y_target = torch.full((n,), target_number, device=device)
y_null = torch.full((n,), NULL_CLASS, device=device)


def v_cfg(x, t):
    # two forward passes of the SAME model, blended; no autograd needed
    with torch.no_grad():
        v_cond = unet(x, t * TIME_SCALE, y_target)
        v_uncond = unet(x, t * TIME_SCALE, y_null)
    return v_uncond + w * (v_cond - v_uncond)

# generating (Heun on the CFG-blended velocity field)
for i in range(steps):
    t_batch = torch.full((n,), i * dt, device=device)
    v1 = v_cfg(x, t_batch)
    v2 = v_cfg(x + v1*dt, t_batch + dt)
    x = x + (v1 + v2)/2 * dt

# plotting
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, im in zip(axes.flat, x.clamp(-1, 1).cpu().numpy()[:, 0]):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
fig.suptitle(f"CFG y={target_number} w={w} from {ckpts[-1]}")
epoch = re.search(r"fm_cfg_(\d+)", ckpts[-1]).group(1)
out_path = f"output/cfg/fm_cfg_w{w}_y{target_number}_steps{steps}_ep{epoch}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")
plt.show()
