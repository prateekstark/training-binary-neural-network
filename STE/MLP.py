import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import *


class BinaryConnect(nn.Module):
    def __init__(self, in_features, out_features, num_units = 2048, momentum=0.15, eps=1e-4, drop_prob=0, batch_affine=True):
        super(BinaryConnect, self).__init__()
        self.in_features = in_features
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.dropout4 = nn.Dropout(p=drop_prob)

        self.fc1 = BinaryLinear(in_features, num_units, bias=False)
        self.fc2 = BinaryLinear(num_units, num_units, bias=False)
        self.fc3 = BinaryLinear(num_units, num_units, bias=False)
        self.fc4 = BinaryLinear(num_units, out_features, bias=False)

        self.bn1 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum, affine=batch_affine)
        self.bn2 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum, affine=batch_affine)
        self.bn3 = nn.BatchNorm1d(num_units, eps=eps, momentum=momentum, affine=batch_affine)
        self.bn4 = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum, affine=batch_affine)

    def forward(self, x):
        x = x.view(-1, self.in_features)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = self.dropout2(x)

        x = self.fc2(x)
        x = F.relu(self.bn2(x))
        x = self.dropout3(x)

        x = self.fc3(x)
        x = F.relu((self.bn3(x)))
        x = self.dropout4(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x

if __name__ == '__main__':
    net = BinaryConnect(784, 10)
    print(net)