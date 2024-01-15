import random
import os
import torch

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

from p2p_server.datasets.mnist import load_mnist

class LocalDataset(Dataset):
    def __init__(self, dataset, index_list) -> None:
        self.dataset = dataset
        self.index_list = index_list
        self.split()

    def split(self):
        """
            to save memory, delete the original dataset.
        """
        dataset = [None for _ in range(len(self.index_list))]
        for i, index in enumerate(self.index_list):
            dataset[i] = self.dataset[index]
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.index_list)


class SplitedDataset:
    """
        PS切分数据, 得到如何划分数据的一个索引列表self.index_list
    """
    def __init__(self, dataset, client_number, seed) -> None:
        self.dataset = dataset
        self.client_number = client_number
        self.seed = seed

    def split(self):
        dataset_length = len(self.dataset)
        index = list(range(dataset_length))
        index_list = [[]]*self.client_number
        random.seed(self.seed)
        random.shuffle(index)

        self.piece_length = int(dataset_length/self.client_number)
        start = 0
        for i in range(self.client_number):
            index_list[i] = index[start:start+self.piece_length]
            start += self.piece_length

        return index_list

    def fedavg_split(self):
        train_label = [data[1] for data in self.dataset]
        train_label = torch.tensor(train_label, dtype=torch.int)
        train_sorted_index = torch.argsort(train_label)


def load_dataset(dataset: str, data_dir: str, batch_size: int):
    train_shuffle = True
    test_shuffle = False
    
    if dataset == "mnist":
        # train_dataset, test_dataset = load_mnist(data_dir, indices)
        train_dataset, test_dataset = load_mnist(data_dir)
    elif dataset == "imagenette2":
        # prepare dataset
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), normalize]
        )
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "imagenette2", "train"),
            transform=transform
        )
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "imagenette2", "val"),
            transform=transform
        )
    elif dataset == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # 载入数据集
        train_dataset = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform_train
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
    else:
        raise ValueError
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle
    )
    
    return train_dataset, train_dataloader, test_dataset, test_dataloader
