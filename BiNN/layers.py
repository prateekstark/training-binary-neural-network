import torch
import torch.nn as nn
import torch.nn.functional as F
from BiNN.functions import *


class BinaryTanh(nn.Module):
	def __init__(self):
		super(BinaryTanh, self).__init__()
		self.hardtanh = nn.Hardtanh()

	def forward(self, x):
		output = self.hardtanh(x)
		output = binarize(output)
		return output


class BinaryLinear(nn.Linear):
	def forward(self, x):
		binary_weights = binarize(self.weight)
		return F.linear(input=x, weight=binary_weights, bias=self.bias)


class BinaryConv2D(nn.Conv2d):
	def forward(self, x):
		binary_kernel = binarize(self.weight)
		return F.conv2d(input=x, weight=binary_kernel, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
