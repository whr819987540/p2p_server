import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch import nn

from p2p_server.utils.my_dataset import get_my_dataset


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2保证输入输出尺寸相同
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class TestModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)
        nn.init.kaiming_uniform_(self.linear.weight)
        self.linear.bias.data.fill_(0)
        print(self.linear.weight, self.linear.bias)

    def forward(self, x):
        return self.linear(x)


if __name__ == "__main__":
    from p2p_server.utils.utils import set_seed

    set_seed(0)
    test_model = TestModel()

    # y=2*x+3
    # x = torch.arange(1, 13, dtype=torch.float32).reshape(-1, 1)
    # y = 2*x+3
    trainset = get_my_dataset(train=True)
    testset = get_my_dataset(train=False)
    trainloader = DataLoader(trainset, 4)
    testloader = DataLoader(testset, 2)

    optimizer = torch.optim.SGD(test_model.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    weights = []
    biases = []
    epochs = 100
    for epoch in range(epochs):
        for x, y in trainloader:
            print(x.shape, y.shape)
            y_hat = test_model(x)
            loss = loss_func(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(test_model.linear.weight, test_model.linear.bias)
            weights.append(test_model.linear.weight.data.clone().data[0][0])
            biases.append(test_model.linear.bias.data.clone().data[0])

    plt.clf()
    plt.plot(list(range(len(weights))), weights, label="weight")
    plt.plot(list(range(len(biases))), biases, label="bias")
    plt.legend()

    plt.show()
    plt.savefig("test.jpg")
    # print(weights,biases)
