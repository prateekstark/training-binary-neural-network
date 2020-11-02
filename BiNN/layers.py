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
		if not hasattr(self.weight, 'latent_'):
			'''
			Initialization
			'''
			self.weight.latent_ = self.weight.data
		self.weight.data = binarize(self.weight.latent_)
		if not self.bias is None:
			self.bias.latent_ = self.bias.data.clone()

		return F.linear(input=x, weight=self.weight, bias=self.bias)



class BinaryConv2D(nn.Conv2d):
	def forward(self, x):
		'''
		Initialization
		'''
		if not hasattr(self.weight, 'latent_'):
			self.weight.latent_ = self.weight.data
		self.weight.data = binarize(self.weight.latent_)
		if not self.bias is None:
			self.bias.latent_ = self.bias.data.clone()

		return F.conv2d(input=x, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

if __name__ == '__main__':
	tensor = torch.Tensor([1, 2, 3, 4])
	bn_layer1 = BinaryLinear(4, 1)
	print(bn_layer1(tensor))
	binary_tanh = BinaryTanh()
	print(binary_tanh(tensor))