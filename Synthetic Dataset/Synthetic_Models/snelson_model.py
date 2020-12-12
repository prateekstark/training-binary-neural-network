import torch
import torch.nn as nn
import torch.nn.functional as F


class SnelsonMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, output_var=False, bias=True, use_bn=False, learn_bn=True, only_last_bn=True):
        super(SnelsonMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_var = output_var
        self.use_bn = use_bn
        self.only_last_bn = only_last_bn
        self.act = torch.tanh
        bn_momentum = 0.15
        bn_eps = 1e-4
        
        if output_size is not None:
            self.output_size = output_size
        else :
            self.output_size = 1

        self.hidden_layers = nn.ModuleList([nn.Linear(in_size, out_size, bias=bias) for in_size, out_size in zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)])
        self.output_layer = nn.Linear(hidden_sizes[-1], self.output_size, bias=bias)
        self.output_bn = nn.BatchNorm1d(output_size, eps=bn_eps, momentum=bn_momentum, affine=learn_bn)

    def forward(self, x):
        x = x.view(-1,self.input_size)
        out = x
        for i in range(len(self.hidden_layers)):
            out = self.hidden_layers[i](out)
            out = self.act(out)
        z = self.output_layer(out)
        z = self.output_bn(z) 
        return z
        
def adjust_learning_rate(lr_deacy, optimizer, epoch, step=1):
    if epoch>0 and epoch % step ==0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_deacy
    return
