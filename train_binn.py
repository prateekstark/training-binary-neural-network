import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
from BiNNOptimizer.BiNNOptimizer import BiNNOptimizer


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



if __name__ == '__main__':
    net = Network()
    # print(net)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5])])

    train_data = datasets.MNIST('data/mnist/training', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST('data/mnist/testing', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    criterion = nn.CrossEntropyLoss()
    optimizer = BiNNOptimizer(net)

    for epoch in range(5):
        net.train(True)
        # optimizer.step()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            # optimizer.zero_grad(net)
            inputs = inputs.view(64, 784)
            print(inputs.shape)
            
            output = net.forward(inputs)
            print('yes 1')
            loss = criterion(output, labels)
            print('yes 2')
            loss.backward()
            print('yes 3')
            # print(inputs.size)
            def closure():
                optimizer.zero_grad()
                # print('yes in 1')
                output = net.forward(inputs.double())
                # print('yes in 2')
                loss = criterion(output, labels)
                # print('yes in 3')
                return loss, output

            loss, output = optimizer.step(closure)
            print('yes 4')
            output = output[0]
            pred = output.argmax(dim=1, keepdims=True)
            correct = pred.eq(labels.view_as(pred)).sum().item()
            print(correct)



