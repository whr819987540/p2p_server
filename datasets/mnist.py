from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import random
import os
import torch
import urllib.request
import gzip
import numpy as np
import PIL.Image as Image
from torch import distributed as dist


class MnistLocalDataset(Dataset):
    def __init__(self, images, labels, client_id):
        self.images = images
        self.labels = labels.astype(int)
        self.client_id = client_id
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ])

    def __getitem__(self, index):
        img = Image.fromarray(self.images[index].reshape(28, 28), mode='L')
        img = self.transform(img)
        target = self.labels[index]
        return img, target

    def __len__(self):
        return len(self.images)


def get_mnist_data(datadir):
    dataroot = 'http://yann.lecun.com/exdb/mnist/'
    key_file = {
        'train_img': 'train-images-idx3-ubyte.gz',
        'train_label': 'train-labels-idx1-ubyte.gz',
        'test_img': 't10k-images-idx3-ubyte.gz',
        'test_label': 't10k-labels-idx1-ubyte.gz'
    }
    os.makedirs(datadir, exist_ok=True)

    for key, filename in key_file.items():
        if os.path.exists(os.path.join(datadir, filename)):
            print(f"already downloaded : {filename}")
        else:
            urllib.request.urlretrieve(
                os.path.join(dataroot, filename),
                os.path.join(datadir, filename)
            )

    with gzip.open(os.path.join(datadir, key_file["train_img"]), "rb") as f:
        train_img = np.frombuffer(f.read(), np.uint8, offset=16)
    train_img = train_img.reshape(-1, 784)

    with gzip.open(os.path.join(datadir, key_file["train_label"]), "rb") as f:
        train_label = np.frombuffer(f.read(), np.uint8, offset=8)

    with gzip.open(os.path.join(datadir, key_file["test_img"]), "rb") as f:
        test_img = np.frombuffer(f.read(), np.uint8, offset=16)
    test_img = test_img.reshape(-1, 784)

    with gzip.open(os.path.join(datadir, key_file["test_label"]), "rb") as f:
        test_label = np.frombuffer(f.read(), np.uint8, offset=8)

    return train_img, train_label, test_img, test_label


def load_mnist(data_dir: str, args):
    fed = True
    RANK, WORLD_SIZE = dist.get_rank(), dist.get_world_size()
    if not fed:
        transform = transforms.ToTensor()
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform)
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform)
    else:    
        iid = args.mnist_iid
        shard_size = args.shard_size
        num_clients = WORLD_SIZE - 1
        dtype = torch.int
        
        train_img, train_label, test_img, test_label = get_mnist_data(data_dir)
        train_sorted_index = np.argsort(train_label)
        train_img = train_img[train_sorted_index]
        train_label = train_label[train_sorted_index]

        if iid:
            random.shuffle(train_sorted_index)
            train_img = train_img[train_sorted_index]
            train_label = train_label[train_sorted_index]

        shard_start_index = [i for i in range(0, len(train_img), shard_size)]
        num_shards = len(shard_start_index) // num_clients
        if RANK == 0:
            random.shuffle(shard_start_index)
            print(f"shard_start_index: {shard_start_index}")
            print(
                f"divide data into {len(shard_start_index)} shards of size {shard_size}"
            )
            
            indices = [torch.empty(0, dtype=dtype) for _ in range(num_clients)]
            for client_id in range(num_clients):
                _index = num_shards * client_id
                for i in range(num_shards):
                    indices[client_id] = torch.cat(
                        (
                            indices[client_id],
                            torch.arange(
                                shard_start_index[_index+i],
                                shard_start_index[_index+i] + shard_size,
                                dtype=dtype
                            )
                        ),
                        dim=0
                    )
                
            server_index = torch.zeros_like(indices[0])
            indices.insert(0, server_index)
            dist.scatter(server_index, indices, src=0)
            print(f"send indices: {indices}")
            train_dataset = MnistLocalDataset(train_img, train_label, client_id)
        else:
            index = torch.zeros(shard_size*num_shards, dtype=torch.int)
            dist.scatter(index, None, src=0)
            index = index.numpy()
            train_dataset = MnistLocalDataset(train_img[index], train_label[index], RANK)

            print(index)

        test_sorted_index = np.argsort(test_label)
        test_img = test_img[test_sorted_index]
        test_label = test_label[test_sorted_index]

        test_dataset = MnistLocalDataset(test_img, test_label, client_id=-1)
        print(len(train_dataset), len(test_dataset))

    return train_dataset, test_dataset

