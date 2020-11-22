import json
import torch
import logging
import dataloader
from torchsummary import summary
from Trainer import BayesianTrainer
from BayesBiNN.models.MLP import BinaryConnect
from BayesBiNN.utils import permute_image
import numpy as np
from tqdm import tqdm


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

    permutations = []
    num_tasks = 5
    for _ in range(num_tasks):
        permutations.append(torch.Tensor(np.random.permutation(784)).long())

    trainer = BayesianTrainer(
        net,
        config["criterion"],
        config["lr_scheduler"],
        logger,
        dataset.get_trainsize(),
        log_params=True,
    )

    for task in range(num_tasks):
        if not task == 0:
            trainer.optim.state["lambda_prior"] = trainer.optim.state["lambda"]
        permutation = permutations[task]

        for param_group in trainer.optim.param_groups:
            param_group["lr"] = 0.001

        for epoch in range(config["epochs"]):
            net.train(True)
            logger.info("starting epoch {}".format(epoch))
            predictions = []
            for inputs, labels in tqdm(trainloader):
                inputs = permute_image(
                    inputs, permutations[task], config["batch_size"], 1
                )
                loss, correct = trainer.train_step(inputs, labels, device=device)
                predictions.append(correct)
            logger.info("train correct: {}".format(sum(predictions) / len(predictions)))
            trainer.lr_scheduler.step()

            for test_task in range(task + 1):
                # assert M >= 0
                net.eval()
                with torch.no_grad():
                    predictions = []
                    for inputs, labels in testloader:
                        inputs = permute_image(
                            inputs, permutations[task], config["batch_size"], 1
                        )
                        loss, correct = trainer.evaluate_step(
                            inputs, labels, device=device
                        )
                        predictions.append(correct)
                    logger.info("{}".format(sum(predictions) / len(predictions)))
