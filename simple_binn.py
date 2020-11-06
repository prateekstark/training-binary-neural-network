import torch
import torch.nn as nn
import torch.nn.functional as F
from STE.MLP import BinaryConnect
from dataloader import Dataset
import logging
from torchsummary import summary
from tqdm import tqdm

"""
Pointers: 
	1) We need to make sure that while we use batch_norm layer, we must use a batch size of more than 1.
"""

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    dataset = Dataset("mnist", data_augmentation=False, validation_split=0.1)
    trainloader, testloader, valloader = dataset.get_dataloaders(batch_size=100)
    net = BinaryConnect(784, 10).to(device)

    summary(net, input_size=(100, 784), device="cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    epochs = 100

    milestones = [int(epochs / 2), int(epochs / 12), 150, 250, 350, 450]

    for epoch in range(epochs):
        result = []
        for inputs, labels in tqdm(trainloader):
            optimizer.zero_grad()
            output = net(inputs.to(device))
            loss = criterion(output, labels.to(device))
            loss.backward()
            for p in net.parameters():
                if hasattr(p, "latent_"):
                    p.data.copy_(p.latent_)
            optimizer.step()
            for p in net.parameters():
                if hasattr(p, "latent_"):
                    p.latent_.copy_(p.data)
            pred = output.argmax(dim=1, keepdims=True)
            correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
            # print(correct)
            result.append(correct)
        print(sum(result) / len(result))
