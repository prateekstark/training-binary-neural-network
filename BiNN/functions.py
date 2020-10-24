import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class Binarize(Function):
    @staticmethod
    def forward(cxt, input):
        output = input.new(input.size())
        output[input >= 0] = 1
        output[input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        return grad_output.clone()

binarize = Binarize.apply

if __name__ == '__main__':
    tensor = torch.Tensor([1, -2, 3, -4, 5])
    print(tensor, binarize(tensor))