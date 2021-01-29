import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader


class TissueSegmentation(Dataset):
    def __init__(
        self,
        image_dir="data/train_imgs",
        labels_dir="data/train_labels",
        input_img_size=(572, 572),
        output_img_size=(388, 388),
        print_dataset=False,
    ):
        X = []
        y = []
        for root, directories, files in os.walk(image_dir, topdown=False):
            for name in files:
                if ".png" in name:
                    X.append(os.path.join(root, name))

        for root, directories, files in os.walk(labels_dir, topdown=False):
            for name in files:
                if ".png" in name:
                    y.append(os.path.join(root, name))

        assert len(X) == len(y)
        X.sort()
        y.sort()
        self.samples = list(zip(X, y))
        del X, y
        if print_dataset:
            self.print_dataset()

        self.input_img_size = input_img_size
        self.output_img_size = output_img_size

    def __len__(self):
        return len(self.samples)

    def print_dataset(self):
        for X, y in self.samples:
            print(X, y)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) / 255.0
        image = cv2.resize(image, self.input_img_size, interpolation=cv2.INTER_NEAREST)
        image = torch.Tensor(image).view(1, 572, 572)

        labels = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255.0
        labels = cv2.resize(
            labels, self.output_img_size, interpolation=cv2.INTER_NEAREST
        )
        labels = torch.Tensor(labels).long()
        # print(image.shape, labels.shape)
        return image, labels


def get_dataloader(
    image_dir="data/train_imgs",
    labels_dir="data/train_labels",
    print_dataset=False,
    batch_size=8,
    input_img_size=(572, 572),
    output_img_size=(388, 388),
):
    dataset = TissueSegmentation(
        image_dir=image_dir,
        labels_dir=labels_dir,
        print_dataset=print_dataset,
        input_img_size=input_img_size,
        output_img_size=output_img_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    dataloader = get_dataloader(
        image_dir="data/train_imgs",
        labels_dir="data/train_labels",
        print_dataset=True,
        batch_size=8,
        input_img_size=(572, 572),
        output_img_size=(388, 388),
    )
