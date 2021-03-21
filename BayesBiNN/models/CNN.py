import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBinaryConnect(nn.Module):
    """VGG-like net used for Cifar10.
    This model is the Conv architecture used in paper "An empirical study of Binary NN optimization".
    """

    def __init__(
        self, in_features, out_features, eps=1e-5, momentum=0.2, batch_affine=False
    ):
        super(VGGBinaryConnect, self).__init__()
        self.in_features = in_features
        self.conv1 = nn.Conv2d(in_features, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128, eps=eps, momentum=momentum, affine=batch_affine)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, eps=eps, momentum=momentum, affine=batch_affine)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, eps=eps, momentum=momentum, affine=batch_affine)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, eps=eps, momentum=momentum, affine=batch_affine)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512, eps=eps, momentum=momentum, affine=batch_affine)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512, eps=eps, momentum=momentum, affine=batch_affine)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024, affine=batch_affine)

        self.fc2 = nn.Linear(1024, 1024, bias=False)
        self.bn8 = nn.BatchNorm1d(1024, affine=batch_affine)

        self.fc3 = nn.Linear(1024, out_features, bias=False)
        self.bn9 = nn.BatchNorm1d(out_features, affine=batch_affine)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn2(x))

        x = F.relu(self.bn3(self.conv3(x)))

        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn4(x))

        x = F.relu(self.bn5(self.conv5(x)))

        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.bn6(x))

        x = x.view(-1, 512 * 4 * 4)

        x = self.fc1(x)
        x = F.relu(self.bn7(x))

        x = self.fc2(x)
        x = F.relu(self.bn8(x))

        x = self.fc3(x)
        x = self.bn9(x)

        return x


class LRNetSmall(nn.Module):
    def __init__(self, input_channels, output_shape, momentum=0.2, batch_affine=False):
        super(LRNetSmall, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(32, momentum=momentum, affine=batch_affine)

        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2, bias=False
        )
        self.bn2 = nn.BatchNorm2d(64, momentum=momentum, affine=batch_affine)

        self.fc1 = nn.Linear(64 * 7 * 7, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512, affine=batch_affine)

        self.fc2 = nn.Linear(512, output_shape, bias=False)
        self.bn4 = nn.BatchNorm1d(output_shape, affine=batch_affine)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2, 2)
        x = F.relu(self.bn1(x))

        x = F.max_pool2d(self.conv2(x), 2, 2)
        x = F.relu(self.bn2(x))

        x = x.view(x.shape[0], -1)
        x = self.bn3(self.fc1(x))
        x = F.relu(x)

        x = self.bn4(self.fc2(x))
        return x


class LRNetBig(nn.Module):
    def __init__(self, input_channels, output_shape, momentum=0.2, batch_affine=False):
        super(LRNetBig, self).__init__()
        self.conv1 = nn.Conv2d(
            input_channels, 128, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(128, momentum=momentum, affine=batch_affine)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=momentum, affine=batch_affine)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=momentum, affine=batch_affine)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, momentum=momentum, affine=batch_affine)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(512, momentum=momentum, affine=batch_affine)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(512, momentum=momentum, affine=batch_affine)

        self.fc1 = nn.Linear(512 * 4 * 4, 1024, bias=False)
        self.bn7 = nn.BatchNorm1d(1024, affine=batch_affine)

        self.fc2 = nn.Linear(1024, 10, bias=False)
        self.bn8 = nn.BatchNorm1d(10, affine=batch_affine)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(x)

        x = x.view(x.shape[0], -1)

        x = self.bn7(self.fc1(x))
        x = F.relu(x)

        x = self.bn8(self.fc2(x))

        return x


if __name__ == "__main__":
    model = LRNetSmall(1, 10)
    X = torch.randn(50, 1, 28, 28)
    y = model(X)
    print(y.shape)

    model = LRNetBig(3, 10)
    X = torch.randn(50, 3, 32, 32)
    y = model(X)
    print(y.shape)
