import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryConnect(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        num_units=2048,
        momentum=0.15,
        eps=1e-4,
        drop_prob=0,
        batch_affine=False,
    ):

        super(BinaryConnect, self).__init__()

        self.in_features = in_features

        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.dropout4 = nn.Dropout(p=drop_prob)

        self.fc1 = nn.Linear(in_features, num_units, bias=False)
        self.fc2 = nn.Linear(num_units, num_units, bias=False)
        self.fc3 = nn.Linear(num_units, num_units, bias=False)
        self.fc4 = nn.Linear(num_units, out_features, bias=False)

        self.bn1 = nn.BatchNorm1d(
            num_units, eps=eps, momentum=momentum, affine=batch_affine
        )
        self.bn2 = nn.BatchNorm1d(
            num_units, eps=eps, momentum=momentum, affine=batch_affine
        )
        self.bn3 = nn.BatchNorm1d(
            num_units, eps=eps, momentum=momentum, affine=batch_affine
        )
        self.bn4 = nn.BatchNorm1d(
            out_features, eps=eps, momentum=momentum, affine=batch_affine
        )

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
        x = F.relu(self.bn3(x))
        x = self.dropout4(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x


class SimpleBinaryConnect(nn.Module):
    def __init__(
        self,
        input_features,
        output_features,
        momentum=0.15,
        batch_affine=True,
        eps=1e-4,
    ):
        super(SimpleBinaryConnect, self).__init__()

        self.input_features = input_features
        self.fc1 = nn.Linear(input_features, 100, bias=False)
        self.bn1 = nn.BatchNorm1d(100, affine=batch_affine, momentum=momentum, eps=eps)

        self.fc2 = nn.Linear(100, 100, bias=False)
        self.bn2 = nn.BatchNorm1d(100, affine=batch_affine, momentum=momentum, eps=eps)

        self.final_layer = nn.Linear(100, output_features, bias=False)
        self.bn3 = nn.BatchNorm1d(
            output_features, affine=batch_affine, momentum=momentum, eps=eps
        )

    def forward(self, x):
        x = x.view(-1, self.input_features)
        x = self.bn1(self.fc1(x))
        x = F.relu(x)

        x = self.bn2(self.fc2(x))
        x = F.relu(x)

        x = self.bn3(self.final_layer(x))
        return x
