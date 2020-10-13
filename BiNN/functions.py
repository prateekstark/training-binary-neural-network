import torch
import torch.nn as nn
import torch.nn.functional as F


class Binarize(torch.autograd.Function):
	@staticmethod
	def __init__(cxt, input_val):
		output = input_val.new(input_val.size())
		output[input_val >= 0] = 1
		output[input_val < 0] = -1
		return output

	@staticmethod
	def forward(cxt, grad_output):
		return grad_output.clone()

binarize = Binarize.apply


