import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for shared modules
import glob, re, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join
from utils import MnistDataloader
from ddpm.ddpm_unet import UNet
from cg.classifier_net import Classifier
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="Train a flow-matching model on MNIST")
parser.add_argument("--target_number", type=int, default=8, help="number to draw")
parser.add_argument("--guidance_lambda", type=float, default=10.0, help="guidance weight")
args = parser.parse_args()
target_number = args.target_number
k_guidance = args.guidance_lambda

device = torch.device("mps")
TIME_SCALE = 1000.0   # must match flow_matching.py

input_path = './input'
(_, _), (x_test, _) = MnistDataloader(
    join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
    join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
    join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
    join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')).load_data()
# x_test = torch.from_numpy(x_test[:1024]).float().to(device)[:, None] / 127.5 - 1

# Load U-Net
unet_ckpts = sorted(glob.glob("checkpoints/fm_v2/fm_*.pth"),
               key=lambda p: int(re.search(r"fm_(\d+)", p).group(1)))
unet = UNet().to(device)
unet.load_state_dict(torch.load(unet_ckpts[-1], map_location=device)["ema"])
unet.eval()

# Load Classifier
clsfy_ckpts = sorted(glob.glob("checkpoints/fm_clf/fm_clf_*.pth"),
               key=lambda p: int(re.search(r"fm_clf_(\d+)", p).group(1)))
classifier = Classifier().to(device)
classifier.load_state_dict(torch.load(clsfy_ckpts[-1], map_location=device))
classifier.eval()

n, steps = 16, 100
dt = 1.0 / steps

x = torch.randn(n, 1, 28, 28, device=device)
y_target = torch.full((n,), target_number, device=device)


def v_conditional(x ,t):
    # unconditional velocity
    with torch.no_grad():
        v = unet(x,t*TIME_SCALE)

    # get classifier gradient
    x_in = x.detach().requires_grad_(True)
    logits = classifier(x_in, t * TIME_SCALE)
    logp = F.log_softmax(logits, dim=1)
    logp = logp[range(n), y_target].sum() 
            # for each of the n images generated, pull out 
            # the log prob of the logit representing the target number
            # and then sum it so the gradient is on a scalar
    grad = torch.autograd.grad(logp, x_in)[0] # grad log p (y|x_t)

    factor = ((1-t)/torch.clamp(t, min=1e-3)).view(-1, 1, 1, 1)
    return v + k_guidance * factor * grad

# generating
for i in range(steps):
    t = i * dt
    t_batch = torch.full((n,), t, device=device)
    v1 = v_conditional(x, t_batch)
    v2 = v_conditional(x + v1*dt, t_batch+dt)
    x = x + (v1 + v2)/2 * dt
    x = x.detach()

# plotting
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for ax, im in zip(axes.flat, x.clamp(-1, 1).cpu().numpy()[:, 0]):
    ax.imshow(im, cmap='gray')
    ax.axis('off')
fig.suptitle(f"samples from {unet_ckpts[-1]}")
epoch = re.search(r"fm_(\d+)", unet_ckpts[-1]).group(1)
out_path = f"output/cg/fm_cg_lambda{k_guidance}_y{target_number}_steps{steps}_ep{epoch}.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=120)
print(f"saved {out_path}")
plt.show()