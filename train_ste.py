import os
import sys
import json
import torch
import logging
from Trainer import STETrainer
from dataloader import Dataset
from torchsummary import summary

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

    wandb_support = False
    if len(sys.argv) == 2:
        config_filename = sys.argv[1]
    elif len(sys.argv) == 3:
        config_filename = sys.argv[1]
        if sys.argv[2] == "wandb_on":
            import wandb

            wandb_support = True
        else:
            raise Exception("Wrong keyword for wandb support!")
    else:
        raise Exception("Wrong Arguments")

    if wandb_support:
        wandb.init(project="bayes-binn")

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Code running on {}".format(device))
    with open(config_filename) as f:
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

    if config["model_architecture"] == "BinaryConnect":
        from STE.models import BinaryConnect

        net = BinaryConnect(
            config["input_shape"],
            config["output_shape"],
            drop_prob=config["drop_prob"],
            batch_affine=config["batch_affine"],
            momentum=config["momentum"],
        ).to(device)
        summary(
            net,
            input_size=(config["batch_size"], config["input_shape"]),
            device=("cuda" if "cuda" in device else "cpu"),
        )
    elif config["model_architecture"] == "VGGBinaryConnect":
        from STE.models import VGGBinaryConnect

        net = VGGBinaryConnect(
            config["input_shape"],
            config["output_shape"],
            momentum=config["momentum"],
            batch_affine=config["batch_affine"],
        ).to(device)

        summary(
            net,
            input_size=(config["input_shape"], 32, 32),
            device=("cuda" if "cuda" in device else "cpu"),
        )
    else:
        raise Exception("Model Architecture NOT supported!")

    logger.info(net)

    if wandb_support:
        wandb.watch(net)

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
        wandb_logger=wandb_support,
        grad_clip_value=config["grad_clip_value"],
        weight_clip_value=config["weight_clip_value"],
    )
