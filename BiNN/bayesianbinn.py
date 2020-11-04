import torch
from optim import BayesBiNNOptimizer


class BayesianBiNN(object):
    def __init__(self, architecture, trainloader, testloader, valloader):
        self.architecture = architecture
        self.trainloader

    def train_step(self):
        self.architecture.train(True)

