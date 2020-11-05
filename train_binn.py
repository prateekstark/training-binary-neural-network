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

if __name__ == "__main__":
    if(torch.cuda.is_available()):
        device = 'cuda:0'
    else:
        device = 'cpu'
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    config = ConfigParser()
    config.read("mnist_config.ini")

    dataset = dataloader.Dataset('mnist', data_augmentation=False, validation_split=0.1)
    trainloader, testloader, valloader = dataset.get_dataloaders(batch_size=100)
    net = BinaryConnect(784, 10).to(device)
    logger.info(net)
    summary(net, input_size=(100, 784), device='cuda')

    print(config["PARAMETERS"]["criterion"])

    if config["PARAMETERS"]["criterion"] == "crossentropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("No such criterion is present!")

    if config["PARAMETERS"]["optimizer"] == "bayesbinn":
        optimizer = BiNNOptimizer(net, train_set_size=dataset.get_trainsize())
    else:
        raise ValueError("Wrong optimizer name, please check!")

    epochs = 100
    milestones = [int(epochs / 2), int(epochs / 12), 150, 250, 350, 450]
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=500, eta_min=1e-16, last_epoch=-1
    )

    logger.info("Code running on {}".format(device))
    logger.info("criterion: {}".format(criterion))
    logger.info("optimizer: {}".format(optimizer))

    for epoch in range(epochs):
        net.train(True)
        logger.info("starting epoch {}".format(epoch))
        predictions = []
        for inputs, labels in tqdm(trainloader):
            def closure():
                optimizer.zero_grad()
                output = net.forward(inputs.to(device))
                loss = criterion(output, labels.to(device))
                # logger.info("loss step: {}".format(loss))
                return loss, output

            loss, output = optimizer.step(closure)
            output = output[0]
            pred = output.argmax(dim=1, keepdims=True)
            correct = pred.eq(labels.to(device).view_as(pred)).sum().item()
            predictions.append(correct)
        logger.info("train correct: {}".format(sum(predictions) / len(predictions)))
        lr_scheduler.step()

        '''
        To check the validation set.
        '''
        
        result = []
        with torch.no_grad():
            for inputs, labels in testloader:
                # raw_noises = []
                mean_vector = torch.where(optimizer.state['mu'] <= 0, torch.zeros_like(optimizer.state['mu']), torch.ones_like(optimizer.state['mu']))
                # raw_noises.append(mean_vector)
                params = optimizer.param_groups[0]['params']
                predictions = []
                raw_noise = mean_vector
                # for raw_noise in raw_noises:
                vector_to_parameters(2 * raw_noise - 1, params)
                output = net.forward(inputs.to(device))
                predictions.append(output)
                prob_tensor = torch.stack(predictions, dim=2)
                probs = torch.mean(prob_tensor, dim=2)
                _, pred_class = torch.max(probs, 1)
                correct = pred_class.eq(labels.to(device).view_as(pred_class)).sum().item()
                result.append(correct)
        logger.info("test correct: {}".format(sum(result) / len(result)))
