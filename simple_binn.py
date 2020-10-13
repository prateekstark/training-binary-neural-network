import torch
import torch.nn as nn
import torch.nn.functional as F
from BiNN.layers import BinaryLinear

'''
Pointers: 
	1) We need to make sure that while we use batch_norm layer, we must use a batch size of more than 1.
'''


class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.binary_fc1 = BinaryLinear(in_features=784, out_features=300, bias=False)
		self.bn_layer1 = nn.BatchNorm1d(300)
		self.output_layer = BinaryLinear(in_features=300, out_features=10, bias=False)
		self.output_bn_layer = nn.BatchNorm1d(num_features=10)

	def forward(self, x):
		x = self.binary_fc1(x)
		x = self.bn_layer1(x)
		x = self.output_layer(x)
		return self.output_bn_layer(x)


if __name__ == '__main__':
	net = Network()
	input_val = torch.randn(2, 784)
	print(net(input_val))
