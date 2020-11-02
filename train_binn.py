import torch
import logging
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from BiNN.models.MLP import BinaryConnect
from BiNN.optim.BayesBiNNOptimizer import BiNNOptimizer


if __name__ == '__main__':
    logging.basicConfig(filename='logfile.log', format="%(levelname)s %(asctime)s %(message)s", filemode="w")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5])])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5,], [0.5])])

    train_data = datasets.MNIST('data/mnist/training', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST('data/mnist/testing', train=False, download=True, transform=test_transform)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    net = BinaryConnect(784, 10)
    logger.info(net)
    summary(net, input_size=(1, 784), device='cpu')

    criterion = nn.CrossEntropyLoss()
    optimizer = BiNNOptimizer(net)

    for epoch in range(5):
        net.train(True)
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.view(64, 784)            
            output = net.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            def closure():
                optimizer.zero_grad()
                output = net.forward(inputs)
                loss = criterion(output, labels)
                return loss, output

            loss, output = optimizer.step(closure)
            output = output[0]
            pred = output.argmax(dim=1, keepdims=True)
            correct = pred.eq(labels.view_as(pred)).sum().item()
            print(correct)
