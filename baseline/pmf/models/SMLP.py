""" simplex constrained quantized classifier model: 1-hidden layer MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import SimplexLayers as sl


class SMLP(nn.Module):
    """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (dataset_size, input_dim)
    Returns:
      y: a tensor of shape (dataset_size, output_dim), with values
      equal to the logits of classifying the digit into one of output_dim classes
    """

    def __init__(self, input_dim, hidden_dim, output_dim, Q_l):
        super(SMLP, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w1 = sl.SLinear(input_dim, hidden_dim, Q_l)
        self.relu1 = nn.ReLU(inplace=True)
        self.w2 = sl.SLinear(hidden_dim, output_dim, Q_l)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.w1(x)
        x = self.relu1(x)
        x = self.w2(x)
        return x


class BinaryConnect(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        Q_l,
        num_units=2048,
        momentum=0.15,
        eps=1e-4,
        drop_prob=0.2,
        batch_affine=True,
    ):
        super(BinaryConnect, self).__init__()
        self.Q_l = Q_l
        self.qlevels = Q_l.size(0)
        self.in_features = in_features
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.dropout4 = nn.Dropout(p=drop_prob)

        self.fc1 = sl.SLinear(in_features, num_units, Q_l, bias=False)
        self.fc2 = sl.SLinear(num_units, num_units, Q_l, bias=False)
        self.fc3 = sl.SLinear(num_units, num_units, Q_l, bias=False)
        self.fc4 = sl.SLinear(num_units, out_features, Q_l, bias=False)

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
        x = F.relu((self.bn3(x)))
        x = self.dropout4(x)

        x = self.fc4(x)
        x = self.bn4(x)

        return x
