import glob, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join
from utils import MnistDataloader
from ddpm_unet import UNet

device = torch.device("mps")
T = 1000
betas = torch.from_numpy(np.linspace(1e-4, 0.02, T)).float().to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, 0)

input_path = './input'
(_, _), (x_test, _) = MnistDataloader(
    join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
    join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
    join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
    join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')).load_data()
x_test = torch.from_numpy(x_test[:1024]).float().to(device)[:, None] / 127.5 - 1

ckpts = sorted(glob.glob("ddpm_*.pth"), key=lambda p: int(re.search(r"(\d+)", p).group(1)))
unet = UNet().to(device)

# noise-prediction MSE on held-out test images, per checkpoint
torch.manual_seed(0)  # same (t, eps) draws for every checkpoint so MSEs are comparable
ts = torch.randint(1, T+1, (x_test.shape[0],), device=device)
eps = torch.randn_like(x_test)
ab = alpha_bars[ts-1].view(-1, 1, 1, 1)
x_t = ab.sqrt()*x_test + (1-ab).sqrt()*eps
with torch.no_grad():
    for ckpt in ckpts:
        unet.load_state_dict(torch.load(ckpt, map_location=device))
        unet.eval()
        mse = torch.mean((unet(x_t, ts) - eps)**2).item()
        print(f"{ckpt}: test noise-pred MSE = {mse:.4f}")

# sample a 4x4 grid from the latest checkpoint
n = 16
x = torch.randn(n, 1, 28, 28, device=device)
with torch.no_grad():
    for t in range(T, 0, -1):
        t_batch = torch.full((n,), t, device=device)
        pred = unet(x, t_batch)
        a, ab = alphas[t-1], alpha_bars[t-1]
        x = (x - (1-a)/torch.sqrt(1-ab)*pred) / torch.sqrt(a)
        if t > 1:
            x = x + torch.sqrt(betas[t-1])*torch.randn_like(x)

fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, im in zip(axes.flat, x.clamp(-1, 1).cpu().numpy()[:, 0]):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
fig.suptitle(f"samples from {ckpts[-1]}")
plt.savefig("samples.png", dpi=120)
print("saved samples.png")
plt.show()
