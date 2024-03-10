from torchvision import models
from typing import Union
import torch

from p2p_server.models.lenet import LeNet
from p2p_server.models.cnn import CNN


model_dict = {
    "resnet18": models.resnet18, "resnet34": models.resnet34,"resnet50": models.resnet50, 
    "resnet101": models.resnet101, "resnet152": models.resnet152, 
    "lenet": LeNet,
    "cnn": CNN,
    "shufflenet_v2_x2_0": models.shufflenet_v2_x2_0
}


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
        elif model_name == "cnn":
            model = model_dict[model_name]()
        elif model_name.startswith("shufflenet"):
            model = model_dict[model_name](pretrained=False)
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
        # elif model_name
    else:
        if num_classes:
            model = model(num_classes)
        else:
            model = model()

    return model

