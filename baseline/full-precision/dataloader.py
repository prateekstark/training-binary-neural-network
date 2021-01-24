import torch
import numpy as np
from torchvision import datasets, transforms


class Dataset(object):
    def __init__(
        self,
        data="mnist",
        data_augmentation=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        validation_split=0,
    ):
        assert (
            data == "mnist" or data == "cifar10" or data == "cifar100"
        ), "only cifar10, cifar100 and mnist datasets are supported!"

        assert 0 <= validation_split < 1, "validation_split must lie between [0, 1)"

        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.validation_split = validation_split
        self.train_sampler = None
        self.valid_sampler = None

        datapath = "./data"

        if data == "mnist":
            if data_augmentation:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(28, padding=4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,)),
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
                )
            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            dataset = datasets.MNIST

        if data == "cifar10":
            if data_augmentation:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                        ),
                    ]
                )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            dataset = datasets.CIFAR10

        if data == "cifar100":
            if data_augmentation:
                train_transform = transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                        ),
                    ]
                )
            else:
                train_transform = transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
                        ),
                    ]
                )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
                ]
            )
            dataset = datasets.CIFAR100

        self.train_data = dataset(
            datapath, train=True, download=True, transform=train_transform
        )

        self.test_data = dataset(
            datapath, train=False, download=True, transform=test_transform
        )

        self.val_data = None

        if validation_split > 0:
            self.val_data = dataset(
                datapath, train=True, download=True, transform=test_transform
            )

            training_points = len(self.train_data)
            indices = list(range(training_points))

            np.random.shuffle(indices)
            validation_size = int(np.floor(validation_split * training_points))

            train_indices, valid_indices = (
                indices[validation_size:],
                indices[:validation_size],
            )

            self.trainsize = len(train_indices)

            print("Number of training samples:", len(train_indices))
            print("Number of validation samples:", len(valid_indices))
            self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                train_indices
            )
            self.valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                valid_indices
            )

    def get_dataloaders(self, batch_size):
        test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
        )

        val_loader = None
        train_loader = None

        if self.validation_split > 0:
            train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=batch_size,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                sampler=self.train_sampler,
            )
            val_loader = torch.utils.data.DataLoader(
                self.val_data,
                batch_size=batch_size,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
                sampler=self.valid_sampler,
            )

        else:
            train_loader = torch.utils.data.DataLoader(
                self.train_data,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=self.pin_memory,
                drop_last=self.drop_last,
                num_workers=self.num_workers,
            )
        return train_loader, test_loader, val_loader

    def get_trainsize(self):
        return self.trainsize


if __name__ == "__main__":
    dataset = Dataset("mnist", validation_split=0.2)
    train_loader, test_loader, val_loader = dataset.get_dataloaders(64)
