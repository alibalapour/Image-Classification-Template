import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from pathlib import Path
import PIL.Image as Image
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
import random

from utils import reverse_normalization


class CustomDataset(Dataset):
    def __init__(self, path, image_size, is_train=False, **kwargs):
        self.path = path
        self.paths = list(Path(self.path).glob("*/*.png"))
        self.image_folder = ImageFolder(self.path)
        classes = self.image_folder.classes
        self.idx_to_class = {key: classes[key] for key in range(len(classes))}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}
        self.transform = self.get_transform(image_size, is_train=is_train, **kwargs)

    def __len__(self):
        return len(self.image_folder)

    def load_image(self, index):
        image_path = self.paths[index]
        return Image.open(image_path)

    def __getitem__(self, index):
        image = self.load_image(index)
        cls_name = self.paths[index].parent.name
        cls_idx = self.class_to_idx[cls_name]
        one_hot_cls = torch.zeros(len(self.idx_to_class))
        one_hot_cls[cls_idx] = 1.0
        return self.transform(image), one_hot_cls

    @staticmethod
    def get_transform(image_size, is_train=True, rand_aug=False, rand_aug_num_ops=10, rand_aug_magnitude=2,
                      imagenet_normalize=True, crop_scale=0.8, jitter_param=0.1, erasing_prob=0.15):
        t = []
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if is_train:
            t.append(transforms.RandomResizedCrop(32, scale=(crop_scale, 1.0), ratio=(1.0, 1.0)))
            t.append(transforms.RandomHorizontalFlip(p=0.5))
            if rand_aug:
                t.append(transforms.RandAugment(num_ops=rand_aug_num_ops, magnitude=rand_aug_magnitude))
            t.append(transforms.ColorJitter(jitter_param, jitter_param, jitter_param))
            t.append(transforms.Resize(image_size))
            t.append(transforms.ToTensor())
            if imagenet_normalize:
                t.append(transforms.Normalize(mean, std))
            t.append(transforms.RandomErasing(p=erasing_prob))

        else:
            t.append(transforms.Resize(image_size))
            t.append(transforms.ToTensor())
            if imagenet_normalize:
                t.append(transforms.Normalize(mean, std))

        transform = transforms.Compose(t)
        return transform


def display(dataset, images, sample_idx):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes
                     )
    for idx, (ax, im) in enumerate(zip(grid, images)):
        ax.set_title(dataset.idx_to_class[torch.argmax(dataset[sample_idx[idx]][1], dim=0).item()])
        ax.set_axis_off()
        ax.imshow(im)
    plt.show()


# just for test :)
if __name__ == '__main__':
    dataset = CustomDataset("../../cifar10/test", (256, 256), is_train=True, rand_aug=True)
    sample_idx = random.sample(list(range(len(dataset))), 16)
    images = np.array([np.transpose(reverse_normalization(dataset[i][0]).numpy(), (1, 2, 0)) for i in sample_idx])
    display(dataset, images, sample_idx)
