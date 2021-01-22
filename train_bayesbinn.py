import json
import torch
import logging
import dataloader
from torchsummary import summary
from Trainer import BayesianTrainer
from BayesBiNN.models.MLP import BinaryConnect


if __name__ == "__main__":
    logging.basicConfig(
        filename="logfile.log",
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Code running on {}".format(device))
    with open("mnist_config.json") as f:
        config = json.load(f)

    logger.info("config: {}".format(config))

    dataset = dataloader.Dataset(
        config["dataset"],
        data_augmentation=config["data_augmentation"],
        validation_split=config["validation_split"],
    )

    trainloader, testloader, valloader = dataset.get_dataloaders(
        batch_size=config["batch_size"]
    )

    net = BinaryConnect(config["input_shape"], config["output_shape"]).to(device)
    logger.info(net)
    summary(
        net,
        input_size=(config["batch_size"], config["input_shape"]),
        device=("cuda" if "cuda" in device else "cpu"),
    )

    trainer = BayesianTrainer(
        net,
        config["criterion"],
        config["lr_scheduler"],
        logger,
        dataset.get_trainsize(),
        log_params=True,
    )

    trainer.train(
        config["epochs"],
        trainloader,
        device=device,
        valloader=valloader,
        testloader=testloader,
        M=0,
    )
