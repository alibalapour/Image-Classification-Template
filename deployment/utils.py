from torchvision import transforms
from torch import nn

idx_to_classes = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, downsample=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride,
                               padding=1)
        if downsample:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2))
        else:
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + shortcut
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, n=1):
        super().__init__()
        self.num_classes = num_classes
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        layer1_list = [ResBlock(64, 64, stride=1) for _ in range(2 * n)]
        self.layer1 = nn.Sequential(
            *layer1_list
        )

        layer2_list = [ResBlock(128, 128, stride=1) for _ in range(2 * n - 1)]
        self.layer2 = nn.Sequential(
            ResBlock(64, 128, stride=2, downsample=True),
            *layer2_list
        )

        layer3_list = [ResBlock(256, 256, stride=1) for _ in range(2 * n - 1)]
        self.layer3 = nn.Sequential(
            ResBlock(128, 256, stride=2, downsample=True),
            *layer3_list
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def get_transform(image_size, imagenet_normalize=True):
    t = []
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    t.append(transforms.Resize(image_size))
    t.append(transforms.ToTensor())
    if imagenet_normalize:
        t.append(transforms.Normalize(mean, std))

    transform = transforms.Compose(t)
    return transform
