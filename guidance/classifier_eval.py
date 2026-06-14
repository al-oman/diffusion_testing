import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # put parent on path for utils

import glob, re
import numpy as np
import torch
import matplotlib.pyplot as plt
from os.path import join
from utils import MnistDataloader
from classifier_net import Classifier
import torch.nn.functional as F

device = torch.device("mps")
TIME_SCALE = 1000.0   # must match flow_matching.py

input_path = './input'
(_, _), (x_test, y_test) = MnistDataloader(
    join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
    join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
    join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
    join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')).load_data()
x_tensor = torch.from_numpy(x_test[:1024]).float().to(device)[:, None] / 127.5 - 1
y_tensor = torch.from_numpy(y_test[:1024]).long().to(device)

ckpts = sorted(glob.glob("output/fm_clf/fm_clf_*.pth"),
               key=lambda p: int(re.search(r"fm_clf_(\d+)", p).group(1)))
classifier = Classifier().to(device)

n, steps = 1024, 100
dt = 1.0 / steps

# u_t loss evaluation of checkpoints
torch.manual_seed(0)  # same (t, x_0) draws for every checkpoint so losses are comparable
ts = torch.full((n,), 1, device=device)
t = ts.view(-1, 1, 1, 1)
with torch.no_grad():
    for ckpt in ckpts[-2:]:
        classifier.load_state_dict(torch.load(ckpt, map_location=device))
        classifier.eval()
        logits = classifier(x_tensor, ts*TIME_SCALE)
        loss = F.cross_entropy(logits, y_tensor)
        print(f"{ckpt}: test classify loss = {loss:.4f}")


classifier.load_state_dict(torch.load(ckpts[-1], map_location=device))
classifier.eval()
# generating
with torch.no_grad():
    logits = classifier(x_tensor, ts*TIME_SCALE)
    loss = F.cross_entropy(logits, y_tensor)
    acc = (logits.argmax(1) == y_tensor).float().mean().item()
    print(f"loss {loss.item():.4f} acc {acc:.3f}")


# # plotting
# fig, axes = plt.subplots(4, 4, figsize=(6, 6))
# for ax, im in zip(axes.flat, x.clamp(-1, 1).cpu().numpy()[:, 0]):
#     ax.imshow(im, cmap='gray')
#     ax.axis('off')
# fig.suptitle(f"samples from {ckpts[-1]}")
# plt.savefig("samples.png", dpi=120)
# print("saved samples.png")
# plt.show()