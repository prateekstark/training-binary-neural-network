import os
import json
import torch
import logging
from Trainer import STETrainer
from dataloader import Dataset
from torchsummary import summary
from STE.MLP import BinaryConnect
import wandb

"""
Pointers: 
	1) We need to make sure that while we use batch_norm layer, we must use a batch size of more than 1.
"""

if __name__ == "__main__":
    id_ = 1
    while "logfile_ste_{}.log".format(id_) in os.listdir():
        id_ += 1

    logging.basicConfig(
        filename="logfile_ste_{}.log".format(id_),
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    wandb.init(project="bayes-binn")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Code running on {}".format(device))

    with open("configs/mnist_config_ste.json") as f:
        config = json.load(f)
    logger.info("config: {}".format(config))

    dataset = Dataset(
        config["dataset"],
        data_augmentation=config["data_augmentation"],
        validation_split=config["validation_split"],
    )

    trainloader, testloader, valloader = dataset.get_dataloaders(
        batch_size=config["batch_size"]
    )

    net = BinaryConnect(
        config["input_shape"],
        config["output_shape"],
        drop_prob=config["drop_prob"],
        batch_affine=config["batch_affine"],
    ).to(device)
    logger.info(net)

    wandb.watch(net)

    summary(
        net,
        input_size=(config["batch_size"], config["input_shape"]),
        device=("cuda" if "cuda" in device else "cpu"),
    )

    logger.info("Starting training with STE optimizer...")

    trainer = STETrainer(
        net,
        config["criterion"],
        config["lr_scheduler"],
        logger,
        log_params=True,
        lr_init=config["lr_init"],
        lr_final=config["lr_final"],
    )

    trainer.train(
        config["epochs"],
        trainloader,
        device=device,
        valloader=valloader,
        testloader=testloader,
        wandb_logger=True,
    )
