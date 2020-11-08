import torch
import torch.nn as nn
import torch.nn.functional as F
from BayesBiNN.optim import BiNNOptimizer
from tqdm import tqdm


class BayesianTrainer(object):
    def __init__(
        self, model, criterion, lr_scheduler, logger, train_set_size, log_params=True
    ):
        self.model = model
        if criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("No such criterion is present!")

        self.optim = BiNNOptimizer(self.model, train_set_size=train_set_size)
        self.logger = logger

        if lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=500, eta_min=1e-16, last_epoch=-1
            )
        else:
            raise ValueError("Wrong Scheduler, please check")

        if log_params:
            self.log_params()

    def log_params(self):
        self.logger.info("Criterion: {}".format(self.criterion))
        self.logger.info("Optimizer: {}".format(self.optim))

    def train_step(self, inputs, labels, device="cpu"):
        def closure():
            self.optim.zero_grad()
            output = self.model.forward(inputs.to(device))
            loss = self.criterion(output, labels.to(device))
            return loss, output

        loss, output = self.optim.step(closure)
        output = output[0]
        pred = output.argmax(dim=1, keepdims=True)
        correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
        return loss, correct

    def train(self, epochs, trainloader, device="cpu"):
        for epoch in range(epochs):
            self.model.train(True)
            self.logger.info("starting epoch {}".format(epoch))
            predictions = []
            for inputs, labels in tqdm(trainloader):
                loss, correct = self.train_step(inputs, labels, device=device)
                predictions.append(correct)
            self.logger.info(
                "train correct: {}".format(sum(predictions) / len(predictions))
            )
            self.lr_scheduler.step()


class STETrainer(object):
    def __init__(
        self, model, criterion, lr_scheduler, logger, lr_init=0.001, log_params=True
    ):
        self.model = model
        if criterion == "crossentropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("No such criterion is present!")

        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr_init)
        self.logger = logger

        if lr_scheduler == "cosine":
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=500, eta_min=1e-16, last_epoch=-1
            )
        else:
            raise ValueError("Wrong Scheduler, please check")

        if log_params:
            self.log_params()

    def log_params(self):
        self.logger.info("Criterion: {}".format(self.criterion))
        self.logger.info("Optimizer: {}".format(self.optim))

    def train_step(self, inputs, labels, device="cpu"):
        self.optim.zero_grad()
        output = self.model(inputs.to(device))
        loss = self.criterion(output, labels.to(device))
        loss.backward()

        for p in self.model.parameters():
            if hasattr(p, "latent_"):
                p.data.copy_(p.latent_)

        self.optim.step()

        for p in self.model.parameters():
            if hasattr(p, "latent_"):
                p.latent_.copy_(p.data)

        pred = output.argmax(dim=1, keepdims=True)
        correct = pred.eq(labels.to(device).view_as(pred)).sum().item()

        return loss, correct

    def train(self, epochs, trainloader, device="cpu"):
        for epoch in range(epochs):
            predictions = []
            for inputs, labels in tqdm(trainloader):
                loss, correct = self.train_step(inputs, labels, device=device)
                predictions.append(correct)
            self.logger.info(
                "train correct: {}".format(sum(predictions) / len(predictions))
            )
            self.lr_scheduler.step()
