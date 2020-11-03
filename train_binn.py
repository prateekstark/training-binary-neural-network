import torch
import logging
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from BiNN.models.MLP import BinaryConnect
from BiNN.optim.BayesBiNNOptimizer import BiNNOptimizer
from configparser import ConfigParser


if __name__ == "__main__":
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    config = ConfigParser()
    config.read("mnist_config.ini")

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0.5,
                ],
                [0.5],
            ),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0.5,
                ],
                [0.5],
            ),
        ]
    )

    train_data = datasets.MNIST(
        "data/mnist/training", train=True, download=True, transform=train_transform
    )
    test_data = datasets.MNIST(
        "data/mnist/testing", train=False, download=True, transform=test_transform
    )

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

    net = BinaryConnect(784, 10)
    logger.info(net)
    summary(net, input_size=(1, 784), device="cpu")

    print(config["PARAMETERS"]["criterion"])

    if config["PARAMETERS"]["criterion"] == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("No such criterion is present!")

    if config["PARAMETERS"]["optimizer"] == "bayesbinn":
        optimizer = BiNNOptimizer(net)
    else:
        raise ValueError("Wrong optimizer name, please check!")

    epochs = 12
    milestones = [int(epochs / 2), int(epochs / 12), 150, 250, 350, 450]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
    )

    logger.info("criterion: {}".format(criterion))
    logger.info("optimizer: {}".format(optimizer))

    for epoch in range(5):
        net.train(True)
        lr_scheduler.step()
        logger.info("starting epoch {}".format(epoch))
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.view(64, 784)
            output = net.forward(inputs)
            loss = criterion(output, labels)
            loss.backward()
            logger.info("loss step: {}".format(loss))

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
