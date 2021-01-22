import os
import sys
import json
import torch
import logging
from dataloader import Dataset
from torchsummary import summary
from Trainer import BayesianTrainer


if __name__ == "__main__":
    id_ = 1
    while "logfile_bayesbinn_{}.log".format(id_) in os.listdir():
        id_ += 1

    logging.basicConfig(
        filename="logfile_bayesbinn_{}.log".format(id_),
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
        from BayesBiNN.models.MLP import BinaryConnect

        net = BinaryConnect(
            in_features=config["input_shape"],
            out_features=config["output_shape"],
            drop_prob=config["drop_prob"],
            batch_affine=config["batch_affine"],
        ).to(device)

        summary(
            net,
            input_size=(config["batch_size"], config["input_shape"]),
            device=("cuda" if "cuda" in device else "cpu"),
        )

    elif config["model_architecture"] == "VGGBinaryConnect":
        from STE.models.CNN import VGGBinaryConnect

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

    logger.info("Starting training with BayesBiNN optimizer...")

    trainer = BayesianTrainer(
        model=net,
        criterion=config["criterion"],
        lr_scheduler=config["lr_scheduler"],
        logger=logger,
        train_set_size=dataset.get_trainsize(),
        mc_steps=config["mc_steps"],
        lr_init=config["lr_init"],
        lr_final=config["lr_final"],
        temperature=config["temperature"],
        log_params=True,
    )

    trainer.train(
        epochs=config["epochs"],
        trainloader=trainloader,
        device=device,
        valloader=valloader,
        testloader=testloader,
        M=config["evaluate_steps"],
        wandb_logger=wandb_support,
    )
