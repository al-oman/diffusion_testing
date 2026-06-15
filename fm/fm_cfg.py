import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path for shared modules
from utils import *
import argparse, glob, re
import torch
from torch.utils.data import TensorDataset, DataLoader
from cfg_unet import CFGUNet

parser = argparse.ArgumentParser(description="Train a class-conditional flow-matching model on MNIST (for classifier-free guidance)")
parser.add_argument("--resume", action="store_true", help="continue from the latest checkpoint in checkpoints/fm_cfg")
parser.add_argument("--epochs", type=int, default=128, help="number of epochs to run")
parser.add_argument("--p_uncond", type=float, default=0.1, help="probability of dropping the label to the null token")
args = parser.parse_args()

input_path = './input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

device = torch.device("mps")
TIME_SCALE = 1000.0   # t in [0,1] -> match the integer-timestep range the time embedding was built for
NUM_CLASSES = 10
NULL_CLASS = NUM_CLASSES   # the conditional model's reserved "no class" token

x_tensor = torch.from_numpy(x_train).float().to(device) / 127.5 - 1   # scale to [-1, 1]
x_tensor = x_tensor[:,None,:,:]
y_tensor = torch.from_numpy(y_train).long().to(device)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

ckpt_dir = "checkpoints/fm_cfg"
os.makedirs(ckpt_dir, exist_ok=True)

unet = CFGUNet(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

ema_decay = 0.999
ema = {k: v.detach().clone() for k, v in unet.state_dict().items()}

start_epoch = 0
if args.resume:
    ckpts = sorted(glob.glob(join(ckpt_dir, "fm_cfg_*.pth")), key=lambda p: int(re.search(r"fm_cfg_(\d+)", p).group(1)))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device)
        unet.load_state_dict(ckpt["model"])
        ema = ckpt["ema"]   # restore EMA so it keeps accumulating instead of warming up from scratch
        start_epoch = int(re.search(r"fm_cfg_(\d+)", ckpts[-1]).group(1)) + 1
        print(f"resuming from {ckpts[-1]}")
    else:
        print(f"--resume set but no checkpoints in {ckpt_dir}, starting fresh")

for epoch in range(start_epoch, start_epoch + args.epochs):
    for batch, (x_1s, ys) in enumerate(loader):
        x_0s = torch.randn_like(x_1s)
        ts = torch.sigmoid(torch.randn(x_1s.shape[0], device=device))  # logit-normal t, concentrated mid-trajectory
        t = ts.view(-1, 1, 1, 1)
        x_ts = (1-t)*x_0s + t*x_1s

        # make a mask that is true for args.p_uncond (10%) of the samples
        drop = torch.rand(ys.shape[0], device=device) < args.p_uncond
        # then replace those 10% of samples with the null token (10)
        ys_in = torch.where(drop, torch.full_like(ys, NULL_CLASS), ys)
        u_preds = unet(x_ts, ts * TIME_SCALE, ys_in)
        us = x_1s - x_0s
        loss = torch.mean((u_preds - us)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():   # EMA of weights
            for k, v in unet.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(ema_decay).add_(v, alpha=1-ema_decay)
                else:
                    ema[k].copy_(v)
        if batch % 100 == 0:
            print(f"epoch {epoch} batch {batch} loss {loss.item():.4f}")
    torch.save({"model": unet.state_dict(), "ema": ema}, join(ckpt_dir, f"fm_cfg_{epoch}.pth"))
