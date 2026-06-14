import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # put parent on path for utils

from utils import *
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
from classifier_net import Classifier

input_path = './input'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

device = torch.device("mps")
TIME_SCALE = 1000.0   # t in [0,1] -> match the integer-timestep range the time embedding was built for

x_tensor = torch.from_numpy(x_train).float().to(device) / 127.5 - 1   # scale to [-1, 1]
x_tensor = x_tensor[:,None,:,:]

y_tensor = torch.from_numpy(y_train).long().to(device)

dataset = TensorDataset(x_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

n_epochs = 128
ckpt_dir = "output/fm_clf"

classifier = Classifier().to(device)
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-4)

for epoch in range(n_epochs):
    for batch, (x_1s, ys) in enumerate(loader):
        x_0s = torch.randn_like(x_1s)
        ts = torch.rand(x_1s.shape[0], device=device)

        t = ts.view(-1, 1, 1, 1)
        x_ts = (1-t)*x_0s + t*x_1s

        logits = classifier(x_ts, ts*TIME_SCALE)

        loss = F.cross_entropy(logits, ys)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            acc = (logits.argmax(1) == ys).float().mean().item()
            print(f"epoch {epoch} batch {batch} loss {loss.item():.4f} acc {acc:.3f}")
    torch.save(classifier.state_dict(), join(ckpt_dir, f"fm_clf_{epoch}.pth"))
        








