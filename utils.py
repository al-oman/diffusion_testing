import numpy as np
import matplotlib.pyplot as plt
import struct
from array import array
from os.path import join

# MNIST Data Loader Class
import struct
from array import array
import numpy as np
import matplotlib.pyplot as plt


class MnistDataloader:
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as f:
            magic, size = struct.unpack(">II", f.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = np.frombuffer(f.read(), dtype=np.uint8).copy()

        with open(images_filepath, 'rb') as f:
            magic, size, rows, cols = struct.unpack(">IIII", f.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(size, rows, cols).copy()

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)

def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure()
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 8);        
        index += 1

def show_image(image):
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(image, cmap=plt.cm.gray)
    plt.show()

def circle(radius=1, x=0, y=0, n=50):
    pts = np.zeros((n,2))
    for i in range(n):
        th = np.random.rand()*2*np.pi
        pts[i,0] = radius*np.cos(th) + x
        pts[i,1] = radius*np.sin(th) + y
    
    return pts

def noisify(arr, noisy_factor):
    noised = arr.copy()
    noised += np.random.normal(0, noisy_factor, size=arr.shape)
    return noised
    
def forward_diffusion_ddpm_step(x, beta, dt):
    return -0.5*beta*x*dt + np.sqrt(beta)*np.random.normal(0, np.sqrt(dt), size=x.shape)

if __name__ == "__main__":
    input_path = './input'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images 
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        noise = np.random.normal(0, 1, x_train.shape[1:])
        noise = np.uint8(noise*256)
        images_2_show.append(noise)
        titles_2_show.append('noise')    



    show_images(images_2_show, titles_2_show)

    plt.show()