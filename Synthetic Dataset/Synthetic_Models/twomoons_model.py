import torch
import torch.nn as nn
import torch.nn.functional as F
from STE_Model.layers import *


class TwoMoonsMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, bias=True, use_bn=False, learn_bn=True, only_last_bn=True):
        super(TwoMoonsMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.use_bn = use_bn
        self.only_last_bn = only_last_bn
        self.output_size = output_size
        self.act = torch.tanh
        bn_momentum = 0.15
        bn_eps = 1e-4

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)

    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.act(out)
        z = self.output_layer(out)

        return z


class TwoMoonsSTE(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, act_func='tanh', output_var=False, bias=True, use_bn=False, learn_bn=True, only_last_bn=True):
        super(TwoMoonsSTE, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_var = output_var
        self.use_bn = use_bn
        self.only_last_bn = only_last_bn
        self.output_size = 1
        self.act = torch.tanh
        bn_momentum = 0.15
        bn_eps = 1e-4


        self.hidden_layers = nn.ModuleList([BinaryLinear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = BinaryLinear(hidden_sizes[-1], self.output_size, bias=bias)
    
    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.act(out)
        z = self.output_layer(out)
        return z

    def predict(self, x):
        logits = self.forward(x)
        prob = torch.sigmoid(logits)
        return prob.reshape(-1).detach().numpy()
