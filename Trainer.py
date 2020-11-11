import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import vector_to_parameters
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

    def evaluate_step(self, inputs, labels, device="cpu", M=0):
        epsilons = []
        for sample in range(M):
            epsilons.append(
                torch.bernoulli(torch.sigmoid(2 * self.optim.state["lambda"]))
            )

        params = self.optim.param_groups[0]["params"]
        if len(epsilons) == 0:
            epsilons.append(
                torch.where(
                    self.optim.state["mu"] <= 0,
                    torch.zeros_like(self.optim.state["mu"]),
                    torch.ones_like(self.optim.state["mu"]),
                )
            )
        output_list = []

        for epsilon in epsilons:
            vector_to_parameters(2 * epsilon - 1, params)
            outputs = self.model(inputs.to(device))
            output_list.append(outputs)

        output_tensor = torch.stack(output_list, dim=2)
        probs = torch.mean(output_tensor, dim=2)
        loss = self.criterion(probs, labels.to(device))
        _, pred = torch.max(probs, 1)
        correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
        return loss, correct

    def evaluate(self, x_loader, M=0, device="cpu", print_info="Evaluation correct"):
        assert M >= 0
        self.model.eval()
        with torch.no_grad():
            predictions = []
            for inputs, labels in x_loader:
                loss, correct = self.evaluate_step(inputs, labels, device=device, M=M)
                predictions.append(correct)
            self.logger.info(
                "{}: {}".format(print_info, sum(predictions) / len(predictions))
            )

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

    def train(
        self, epochs, trainloader, device="cpu", valloader=None, testloader=None, M=0
    ):
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
            if valloader:
                self.evaluate(
                    valloader, device=device, print_info="Validation accuracy", M=M
                )
            if testloader:
                self.evaluate(
                    testloader, device=device, print_info="Test accuracy", M=M
                )


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

    def evaluate_step(self, inputs, labels, device="cpu"):
        output = self.model(inputs.to(device))
        loss = self.criterion(output, labels.to(device))
        pred = output.argmax(dim=1, keepdims=True)
        correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
        return loss, correct

    def evaluate(self, x_loader, device="cpu", print_info="Evaluation correct"):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, labels in x_loader:
                loss, correct = self.evaluate_step(inputs, labels, device=device)
                predictions.append(correct)
            self.logger.info(
                "{}: {}".format(print_info, sum(predictions) / len(predictions))
            )

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

    def train(self, epochs, trainloader, device="cpu", valloader=None, testloader=None):
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
            if valloader:
                self.evaluate(
                    valloader, device=device, print_info="Validation accuracy"
                )
            if testloader:
                self.evaluate(testloader, device=device, print_info="Test accuracy")
