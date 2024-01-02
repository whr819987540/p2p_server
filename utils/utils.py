import argparse
import logging
import random
import torch
import os
import numpy
import json
import logging
import pickle
from scipy import stats

from datetime import datetime
from torchvision import models
from typing import Union
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch import distributed as dist

from p2p_server.rpc.rpc_client import readJsonc, loadConfig, to_namespace
from p2p_server.utils.my_model import LeNet


model_dict = {"resnet18": models.resnet18, "resnet34": models.resnet34,
              "resnet50": models.resnet50, "resnet101": models.resnet101, "resnet152": models.resnet152, "lenet": LeNet}
SIMULATION_MODE = "simulation"
STANDALONE_MODE = "standalone"

def get_args():
    parser = argparse.ArgumentParser(description="Train models on Imagenette under ASGD")
    parser.add_argument("--model", type=str, default="resnet18", help="The job's name.")
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--world_size", type=int, default=3,
                        help="Total number of workers including the parameter server and clients.")
    default_data_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.dirname(default_data_dir)
    default_data_dir = os.path.dirname(default_data_dir)
    default_data_dir = os.path.join(default_data_dir, "data")
    parser.add_argument("--data_dir", type=str,
                        default=default_data_dir, help="The location of dataset.")
    parser.add_argument("--dataset", type=str,
                        default="cifar10", help="Dataset.")
    parser.add_argument("--num_classes", type=int,
                        default=10, help="number of classes in your dataset.")
    parser.add_argument("--master_addr", type=str, default="localhost", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="29600", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size of each worker during training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--seed", type=int, default=0, help="random seed.")
    parser.add_argument("--num_evaluate_threads", type=int, default=1,
                        help="parameter server uses certain number of threads to evaluate the model.")
    parser.add_argument("--log_level", type=str, default="DEBUG",
                        help="debug level: NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL, CRITICAL")
    parser.add_argument("--aggregate_frequency", type=int, default=5,
                        help="aggregate the gradients every certain number of iterations in one epoch")
    parser.add_argument("--mode", type=str, default=SIMULATION_MODE,
                        help="set the running mode. simulation and standalone are available.")

    args = parser.parse_args()

    return args


def get_logger(args, name):
    # 2023-08-02 19:23:44 [0] "/home/whr/fs_gnn/dist_gcn_train.py", line 844, DEBUG: torch.Size([512, 500])

    # note: 如果是多进程，对logging模块的初始化需要在每个进程中都运行一次
    # 并且，logging模块是线程安全的，但并不是进程安全的
    # 如果要保证进程安全，需要将其它进程的消息汇总到一个进程，然后由同一进程中的某些logger（标识进程）来完成
    log_format = f'%(asctime)s %(name)s "%(pathname)s", line %(lineno)d, %(levelname)s: %(message)s\n'

    assert args.log_level in logging._nameToLevel.keys(
    ), "log_level should be one of [NOTSET, DEBUG, INFO, WARNING, ERROR, FATAL, CRITICAL]"

    handler = logging.StreamHandler()
    handler.setLevel(logging._nameToLevel[args.log_level])
    formatter = logging.Formatter(fmt=log_format)
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging._nameToLevel[args.log_level])
    logger.addHandler(handler)

    return logger


def get_updated_config_file(RANK:int, master_addr: str, master_port: int, model: str, dataset: str):
    """
        load dict object from jsonc
        
        update the dict object by params and datetime
        
        return the updated dict object and the path of the updated json file
    """
    current_path = os.path.abspath(__file__)
    current_path = os.path.dirname(current_path)
    current_path = os.path.dirname(current_path)

    jsonc_config_path = os.path.join(current_path, "rpc", "rpc_server", "config.jsonc")
    json_config_path = os.path.join(current_path, "rpc", "rpc_server", "config.json")

    # update config.json according to the datetime
    config = json.loads(readJsonc(jsonc_config_path))
    config['server']['ServerIP'] = master_addr
    config['server']['ServerPort'] = int(master_port)
    config['model']['ModelPath'] = os.path.join(
        config['model']['ModelPath'], model, dataset, get_datetime_str())
    if os.environ.get('StackName'):
        config['tracker']['URLList'] = [
            [
                f"udp://{os.environ['StackName']}_tracker:6969/announce",
            ],
        ]

    return config, json_config_path


def update_config_file(RANK: int, master_addr: str, master_port: int, model: str, dataset: str, logger: logging.Logger):
    if RANK == 0:
        config, json_config_path = get_updated_config_file(RANK, master_addr, master_port, model, dataset)
        with open(json_config_path, 'w') as f:
            f.write(json.dumps(config))

        # DONE: transfer the updated config.json instead of relying on NFS
        config_json = json.dumps(config)
        config_bytes = config_json.encode()
        config_tensor = torch.from_numpy(numpy.frombuffer(config_bytes, numpy.uint8))
        # size
        dist.broadcast(torch.tensor([config_tensor.shape[0]], dtype=torch.int64), 0)
        # data
        dist.broadcast(config_tensor, 0)
        config = to_namespace(config)
    else:
        # size
        config_size = torch.empty(1, dtype=torch.int64)
        dist.broadcast(config_size, 0)
        # data
        config_tensor = torch.empty(config_size[0], dtype=torch.uint8)
        dist.broadcast(config_tensor, 0)
        config_bytes = config_tensor.numpy().tobytes()
        config_json = config_bytes.decode()
        config = json.loads(config_json)
        config = to_namespace(config)

    logger.info(f"config {config}")
    logger.info(
        f"master_addr {config.server.ServerIP}:{config.server.ServerPort}, model_root_dir {config.model.ModelPath}")
    return config


def get_datetime_str():
    return datetime.strftime(datetime.now(), "%Y_%m_%d_%H_%M_%S")


def load_dataset(dataset: str, data_dir: str, batch_size: int,):
    if dataset == "mnist":
        # 定义数据预处理方式
        transform = transforms.ToTensor()

        # 定义训练数据集
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform)
        # 定义训练批处理数据
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,  # for distributedsampler, this shuffle must be False
        )

        # 定义测试数据集
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform)

        # 定义测试批处理数据
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        )
    elif dataset == "imagenette2":
        # prepare dataset
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), normalize]
        )
        train_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "imagenette2", "train"),
            transform=transform
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False)
        test_dataset = datasets.ImageFolder(
            root=os.path.join(data_dir, "imagenette2", "val"),
            transform=transform
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
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

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        test_dataset = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform_test
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        raise ValueError

    return train_dataset, train_dataloader, test_dataset, test_dataloader


def split_dataset(dataset: Dataset, N: int, seed: int):
    dataset_length = len(dataset)
    index = list(range(dataset_length))
    index_list = [[]]*N
    random.seed(seed)
    random.shuffle(index)

    piece_length = int(dataset_length/N)
    start = 0
    datasets = [[]]*N

    for i in range(N):
        index_list[i] = index[start:start+piece_length]
        start += piece_length
        feat = []
        label = []
        for j in index_list[i]:
            tmp = dataset[j]
            if tmp[0].shape[0] == 1:
                feat.append(tmp[0].squeeze())
            else:
                feat.append(tmp[0])
            label.append(tmp[1])
        feat = torch.stack(feat)
        label = torch.tensor(label)
        datasets[i] = [feat, label]

    return datasets


class LocalDataset(Dataset):
    def __init__(self, dataset, index_list) -> None:
        self.dataset = dataset
        self.index_list = index_list

    def __getitem__(self, index):
        return self.dataset[self.index_list[index]]

    def __len__(self):
        return len(self.index_list)


class SplitedDataset:
    """
        PS切分数据, 得到如何划分数据的一个索引列表self.index_list
    """

    def __init__(self, dataset, N, seed) -> None:
        self.dataset = dataset
        self.client_number = N
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


def get_model(model: Union[str, torch.nn.Module], dataset: str, num_classes: int = None):
    if isinstance(model, str):
        model_name = model
        # initialize model parameters
        assert model_name in model_dict.keys(), \
            f'model {model_name} is not in the model list: {list(model_dict.keys())}'

        if model_name.startswith("resnet"):
            model = model_dict[model_name](pretrained=False)
        elif model_name == "lenet":
            model = model_dict[model_name]()
        else:
            raise ValueError

        if model_name.startswith("resnet") and dataset == "cifar10":
            # 修改第一个卷积层,以适应CIFAR-10输入
            model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            # 修改最后一层全连接层,以适应10分类
            if model_name == "resnet18":
                dimension = 512
            elif model_name == "resnet34":
                dimension = 512
            elif model_name == "resnet50":
                dimension = 2048
            elif model_name == "resnet101":
                dimension = 2048
            elif model_name == "resnet152":
                dimension = 2048
            model.fc = torch.nn.Linear(dimension, num_classes)
        elif model_name.startswith("resnet") and dataset == "mnist":
            model.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
    else:
        if num_classes:
            model = model(num_classes)
        else:
            model = model()

    return model


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    numpy.random.seed(seed)


def GetPearsonCorrelationCoefficient():
    data = load_client_profile()
    computation = []
    communication = []
    for value in data.values():
        computation.append(value['computation'])
        communication.append(value['communication'])

    # Pearson correlation coefficient: 0.0024169218848259294
    # P-value: 0.08744723051476527
    correlation, p_value = stats.pearsonr(computation, communication)
    print("Pearson correlation coefficient:", correlation)
    print("P-value:", p_value)


def GetSpearmanCorrelationCoefficient():
    data = load_client_profile()
    computation = []
    communication = []
    for value in data.values():
        computation.append(value['computation'])
        communication.append(value['communication'])

    # Spearman correlation coefficient: 0.003023359711877614
    # P-value: 0.03252991845126464
    correlation, p_value = stats.spearmanr(computation, communication)
    print("Spearman correlation coefficient:", correlation)
    print("P-value:", p_value)


def load_client_profile():
    path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(path, "client_device_capacity")
    with open(path, "rb") as f:
        res = pickle.load(f)

    # # 500000
    # print(len(res))

    # keys = [1, 4,5,7,8,3463,34635]
    # for key in keys:
    #     print(key,res[key])
    # # 1 {'computation': 153.0, 'communication': 2209.615982329485}
    # # 4 {'computation': 149.0, 'communication': 13507.696000153657}
    # # 5 {'computation': 29.0, 'communication': 6924.407283130328}
    # # 7 {'computation': 176.0, 'communication': 32545.573620752573}
    # # 8 {'computation': 44.0, 'communication': 42360.068898122656}
    # # 3463 {'computation': 21.0, 'communication': 11154.383933690891}
    # # 34635 {'computation': 82.0, 'communication': 82504.44631466508}

    # GetPearsonCorrelationCoefficient()
    # # Pearson correlation coefficient: 0.0024169218848259294
    # # P-value: 0.08744723051476527
    # GetSpearmanCorrelationCoefficient()
    # # Pearson correlation coefficient: 0.003023359711877614
    # # P-value: 0.03252991845126464

    return res


if __name__ == "__main__":
    args = get_args()
    print(args)
    logger = get_logger(args, "test")
    logger.info("test")

    train_dataset, train_dataloader, test_dataset, test_dataloader = load_dataset(
        args.dataset, args.data_dir, args.batch_size)

    # print(train_dataset)
    # print(test_dataset)
    # split_datasets = split_dataset(train_dataset, 7, 0)
    # for feature, label in split_datasets:
    #     print(feature.shape, label.shape)

    print(args.batch_size, len(train_dataset))
    i = 0
    for x, y in train_dataloader:
        print(x.shape)
        i += 1
    print(i, int(len(train_dataset)/args.batch_size))
