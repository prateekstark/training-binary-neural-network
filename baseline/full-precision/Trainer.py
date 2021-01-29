import torch
import torch.nn as nn
from torch.nn.utils import vector_to_parameters, clip_grad_value_
from tqdm import tqdm


class BaselineTrainer(object):
    def __init__(
        self,
        model,
        criterion,
        lr_scheduler,
        logger,
        lr_init=0.001,
        lr_final=1e-16,
        log_params=True,
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
                self.optim, T_max=500, eta_min=lr_final, last_epoch=-1, verbose=True
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
        loss = self.criterion(output, labels.to(device)) / labels.shape[0]
        pred = output.argmax(dim=1, keepdims=True)
        correct = (
            pred.eq(labels.to(device).view_as(pred)).sum().item() / labels.shape[0]
        ) * 100
        return loss, correct

    def evaluate(
        self,
        x_loader,
        device="cpu",
        print_info="Evaluation correct",
        wandb_logger=False,
    ):
        if wandb_logger:
            import wandb
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for inputs, labels in x_loader:
                loss, correct = self.evaluate_step(inputs, labels, device=device)
                predictions.append(correct)
        accuracy = sum(predictions) / len(predictions)
        self.logger.info("{}: {}".format(print_info, accuracy))
        if wandb_logger:
            wandb.log({print_info: accuracy})

    def train_step(
        self, inputs, labels, device="cpu", grad_clip_value=1, weight_clip_value=1
    ):
        self.optim.zero_grad()
        output = self.model(inputs.to(device))
        loss = self.criterion(output, labels.to(device))

        loss.backward()
        clip_grad_value_(self.model.parameters(), grad_clip_value)

        self.optim.step()

        for p in self.model.parameters():
            p.data.clamp_(-weight_clip_value, weight_clip_value)

        pred = output.argmax(dim=1, keepdims=True)
        correct = (
            pred.eq(labels.to(device).view_as(pred)).sum().item() / labels.shape[0]
        ) * 100

        return loss, correct

    def train(
        self,
        epochs,
        trainloader,
        device="cpu",
        valloader=None,
        testloader=None,
        wandb_logger=False,
        grad_clip_value=1,
        weight_clip_value=1,
    ):
        if wandb_logger:
            import wandb
        for epoch in range(epochs):
            self.model.train(True)
            self.logger.info("starting epoch {}".format(epoch))
            predictions = []
            losses = []
            for inputs, labels in tqdm(trainloader):
                loss, correct = self.train_step(
                    inputs,
                    labels,
                    device=device,
                    grad_clip_value=grad_clip_value,
                    weight_clip_value=weight_clip_value,
                )
                predictions.append(correct)
                losses.append(loss)
            training_accuracy = sum(predictions) / len(predictions)
            average_loss = sum(losses) / len(losses)

            self.logger.info("train correct: {}".format(training_accuracy))

            if wandb_logger:
                wandb.log(
                    {"Training Accuracy": training_accuracy, "Loss": average_loss}
                )

            self.lr_scheduler.step()

            if valloader:
                self.evaluate(
                    valloader,
                    device=device,
                    print_info="Validation Accuracy",
                    wandb_logger=wandb_logger,
                )
            if testloader:
                self.evaluate(
                    testloader,
                    device=device,
                    print_info="Test Accuracy",
                    wandb_logger=wandb_logger,
                )
