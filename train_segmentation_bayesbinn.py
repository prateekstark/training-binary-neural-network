import os
import torch
import random
import logging
import argparse
from segmentation.prepare_data import get_dataloader
from torchsummary import summary
from BayesBiNN.models.UNET import UNet20
from BayesBiNN.optim import BiNNOptimizer
import matplotlib.pyplot as plt
from tqdm import tqdm
from segmentation.utils import IoULoss


def output_mean(output):
    temp = 0
    length = len(output)
    for i in range(length):
        temp += output[i]
    temp = temp / length
    return temp


if __name__ == "__main__":
    id_ = str(random.randint(0, 100000))
    if not id_ in os.listdir("."):
        os.mkdir(id_)

    logging.basicConfig(
        filename="{}/logfile_bayesbinn.log".format(id_),
        format="%(levelname)s %(asctime)s %(message)s",
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", help="Default=6", type=int, default=6)
    parser.add_argument("--mc-steps", help="Default=1", type=int, default=1)
    parser.add_argument("--lr", help="Default=1e-4", type=float, default=1e-10)
    parser.add_argument("--temperature", help="Default=1e-10", type=float, default=1e-8)
    parser.add_argument(
        "-print_dataset", help="Default=False", action="store_true", default=False
    )
    parser.add_argument("--epochs", help="Default=500", type=int, default=500)
    parser.add_argument("--dropout_rate", help="Default=0.05", type=float, default=0.05)
    args = parser.parse_args()

    logger.info(args)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info("Device: {}".format(device))

    trainloader = get_dataloader(
        image_dir="segmentation/data/train_imgs",
        labels_dir="segmentation/data/train_labels",
        print_dataset=args.print_dataset,
        batch_size=args.batch_size,
        input_img_size=(572, 572),
        output_img_size=(388, 388),
    )

    valloader = get_dataloader(
        image_dir="segmentation/data/val_imgs",
        labels_dir="segmentation/data/val_labels",
        print_dataset=args.print_dataset,
        batch_size=args.batch_size,
        input_img_size=(572, 572),
        output_img_size=(388, 388),
    )

    net = UNet20(dropout_rate=args.dropout_rate).to(device)

    summary(
        net,
        input_size=(1, 572, 572),
        device=("cuda" if "cuda" in device else "cpu"),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optim = BiNNOptimizer(
        net,
        train_set_size=30,
        learning_rate=args.lr,
        temperature=args.temperature,
        N=args.mc_steps,
    )

    test_loss = IoULoss()

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=500, eta_min=1e-16, last_epoch=-1, verbose=True
    )

    for epoch in range(args.epochs):
        total_loss = 0
        net.train(True)
        for inputs, labels in tqdm(trainloader):

            def closure():
                optim.zero_grad()
                output = net(inputs.to(device))
                loss = criterion(output, labels.to(device))
                return loss, output

            loss, output = optim.step(closure)
            total_loss += loss
        total_loss /= len(trainloader)
        logger.info(
            "epoch: {}                        loss: {}".format(epoch, total_loss)
        )
        lr_scheduler.step()
        logger.info(optim.param_groups[0]["lr"])
        mean_output = output_mean(output)
        plt.imshow(torch.argmax(mean_output[0], dim=0).cpu().detach().numpy())
        plt.savefig("{}/output_{}.png".format(id_, epoch))
        plt.imshow(mean_output[0][0].cpu().detach().numpy())
        plt.savefig("{}/output_raw_{}.png".format(id_, epoch))
        net.eval()
        iou_loss = 0
        for inputs, labels in tqdm(valloader):
            output = net(inputs.to(device))
            output = torch.argmax(output, dim=1)
            iou_loss += test_loss(output, labels.to(device))
        logger.info("validation score: {}".format(iou_loss / len(valloader)))
