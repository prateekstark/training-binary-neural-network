import torch
import logging
import torchvision
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from BiNN.models.MLP import BinaryConnect
from BiNN.optim.BayesBiNNOptimizer import BiNNOptimizer
from configparser import ConfigParser
import dataloader
from torch.nn.utils import vector_to_parameters
from tqdm import tqdm
import json
from Trainer import BayesianTrainer

if __name__ == "__main__":
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    logger.info("Code running on {}".format(device))
    with open("mnist_config.json") as f:
        config = json.load(f)

    logger.info("config: {}".format(config))

    dataset = dataloader.Dataset(
        config["dataset"],
        data_augmentation=config["data_augmentation"],
        validation_split=config["validation_split"],
    )

    trainloader, testloader, valloader = dataset.get_dataloaders(batch_size=config["batch_size"])

    net = BinaryConnect(config["input_shape"], config["output_shape"]).to(device)
    logger.info(net)
    summary(net, input_size=(config["batch_size"], config["input_shape"]), device=('cuda' if 'cuda' in device else 'cpu'))

    trainer = BayesianTrainer(net, config['criterion'], config['lr_scheduler'], logger, dataset.get_trainsize(), log_params=True)
    trainer.train(config['epochs'], trainloader, device=device)

'''
    for epoch in range(epochs):
        net.train(True)
        logger.info("starting epoch {}".format(epoch))
        predictions = []
        for inputs, labels in tqdm(trainloader):

            def closure():
                optimizer.zero_grad()
                output = net.forward(inputs.to(device))
                loss = criterion(output, labels.to(device))
                return loss, output

            loss, output = optimizer.step(closure)
            output = output[0]
            pred = output.argmax(dim=1, keepdims=True)
            correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
            predictions.append(correct)
        logger.info("train correct: {}".format(sum(predictions) / len(predictions)))
        lr_scheduler.step()

        """
        To check the validation set.
        """

        result = []
        with torch.no_grad():
            for inputs, labels in testloader:
                mean_vector = torch.where(
                    optimizer.state["mu"] <= 0,
                    torch.zeros_like(optimizer.state["mu"]),
                    torch.ones_like(optimizer.state["mu"]),
                )
                params = optimizer.param_groups[0]["params"]
                predictions = []
                raw_noise = mean_vector
                vector_to_parameters(2 * raw_noise - 1, params)
                output = net.forward(inputs.to(device))
                predictions.append(output)
                prob_tensor = torch.stack(predictions, dim=2)
                probs = torch.mean(prob_tensor, dim=2)
                _, pred_class = torch.max(probs, 1)
                correct = (
                    pred_class.eq(labels.to(device).view_as(pred_class)).sum().item()
                )
                result.append(correct)
        logger.info("test correct: {}".format(sum(result) / len(result)))
'''