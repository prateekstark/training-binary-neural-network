import torch
import torch.nn as nn
import torch.nn.functional as F


class SquaredHingeLoss(nn.Module):
    def __init__(self):
        super(SquaredHingeLoss, self).__init__()
