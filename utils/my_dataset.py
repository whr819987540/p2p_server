from torch.utils.data import Dataset
import torch


def get_my_dataset(n: int = 300, train: bool = True):
    return MyDataset(n)


class MyDataset(Dataset):
    def __init__(self, n: int = 300) -> None:
        super().__init__()
        self.x = torch.rand(n).reshape(-1, 1)
        self.y = 2*self.x+3+torch.normal(0, 0.1, size=(n,)).reshape(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
