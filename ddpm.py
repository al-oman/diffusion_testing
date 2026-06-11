from utils import *
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

T = 1000
beta_1 = 1e-4
beta_T = 0.02

device = torch.device("mps")

betas = np.linspace(beta_1, beta_T, T)
alphas = 1 - betas
alpha_bars = torch.from_numpy(np.cumprod(alphas)).float().to(device)

def x_t_given_x_0(t, x_0s):
    if t.shape[0] != x_0s.shape[0]:
        raise ValueError(
            f"timestep batch size ({t.shape[0]}) does not match image batch size ({x_0s.shape[0]})"
        )
    alpha_bar = alpha_bars[t-1]
    alpha_bar = alpha_bar.view(-1,1,1,1)
    eps = torch.randn_like(x_0s)
    x_ts = torch.sqrt(alpha_bar)*x_0s + torch.sqrt(1 - alpha_bar)*eps
    return x_ts, eps

x_tensor = torch.from_numpy(x_train).float().to(device) / 127.5 - 1   # scale to [-1, 1]
x_tensor = x_tensor[:,None,:,:]

dataset = TensorDataset(x_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

n_epochs = 24

unet = UNet().to(device)
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

for epoch in range(n_epochs):
    for (x_0s,) in loader:
        ts = torch.randint(1, T+1, (x_0s.shape[0],), device=device)
        x_ts, epsilons = x_t_given_x_0(ts, x_0s)
        pred = unet(x_ts, ts)
        loss = torch.mean(torch.square(epsilons - pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('-')
    torch.save(unet.state_dict(), f"ddpm_{epoch}.pth")







