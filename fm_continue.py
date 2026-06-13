from utils import *
import glob, re
import torch
from torch.utils.data import TensorDataset, DataLoader
from ddpm_unet import UNet

input_path = './input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

device = torch.device("mps")

x_tensor = torch.from_numpy(x_train).float().to(device) / 127.5 - 1   # scale to [-1, 1]
x_tensor = x_tensor[:,None,:,:]

dataset = TensorDataset(x_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

n_epochs = 128

unet = UNet().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

ckpt_dir = "output/fm"
ckpts = sorted(glob.glob(join(ckpt_dir, "fm_*.pth")), key=lambda p: int(re.search(r"fm_(\d+)", p).group(1)))
last_epoch = int(re.search(r"fm_(\d+)", ckpts[-1]).group(1))
unet.load_state_dict(torch.load(ckpts[-1], map_location=device))
print(f"resuming from {ckpts[-1]}")

for epoch in range(last_epoch + 1, last_epoch + 1 + n_epochs):
    for batch, (x_1s,) in enumerate(loader):
        x_0s = torch.randn_like(x_1s)
        ts = torch.rand(x_1s.shape[0], device=device)
        t = ts.view(-1, 1, 1, 1)
        x_ts = (1-t)*x_0s + t*x_1s
        u_preds = unet(x_ts, ts)
        us = x_1s - x_0s
        loss = torch.mean((u_preds - us)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            print(f"epoch {epoch} batch {batch} loss {loss.item():.4f}")
    torch.save(unet.state_dict(), join(ckpt_dir, f"fm_{epoch}.pth"))
