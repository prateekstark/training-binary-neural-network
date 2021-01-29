import torch
import torch.nn as nn
from PIL import Image, ImageSequence


def retrieve_data():
    train_images = Image.open("data/train-volume.tif")
    train_labels = Image.open("data/train-labels.tif")
    test_images = Image.open("data/test-volume.tif")

    for i, img in enumerate(ImageSequence.Iterator(train_images)):
        img.save("data/train_imgs/img_{}.png".format(i))

    for i, label in enumerate(ImageSequence.Iterator(train_labels)):
        label.save("data/train_labels/label_{}.png".format(i))

    for i, img in enumerate(ImageSequence.Iterator(test_images)):
        img.save("data/test_imgs/img_{}.png".format(i))


def count_parameters(model):
    from prettytable import PrettyTable

    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return 1 - IoU
