import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def print_params(model):
    for param in model.parameters():
        print(param.grad)


def binarize(model):
    with torch.no_grad():
        for param in model.parameters():
            param.data = torch.sign(param.data)


def assign_grads(from_model, to_model):
    with torch.no_grad():
        for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
            to_param.grad = from_param.grad


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200, bias=False)
        self.fc2 = nn.Linear(200, 100, bias=False)
        self.fc3 = nn.Linear(100, 60, bias=False)
        self.fc4 = nn.Linear(60, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


if __name__ == "__main__":
    num_epochs = 10
    for _ in range(num_epochs):
        net = Network()
        criterion = nn.MSELoss()
        binet = deepcopy(net)
        binarize(binet)
        input_val = torch.randn(1, 784)
        truth = torch.randn(1, 10)
        output_val = binet(input_val)
        print(output_val)

        loss = criterion(output_val, truth)
        binet.zero_grad()
        loss.backward()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        optimizer.zero_grad()
        assign_grads(binet, net)
        optimizer.step()
