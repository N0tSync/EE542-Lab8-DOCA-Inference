"""Two-convolution CNN for Part B.

Layout (for a 3 x 32 x 32 input, e.g. CIFAR-10):
    conv1 3->32, 3x3, pad 1  -> ReLU -> 2x2 max pool   : 32 x 16 x 16
    conv2 32->64, 3x3, pad 1 -> ReLU -> 2x2 max pool   : 64 x  8 x  8
    flatten                                            : 4096
    fc1 4096 -> 512 -> ReLU
    fc2 512 -> 10   -> ReLU (output ReLU, trained with MSE on one-hot targets)

For 1-channel 28 x 28 input (MNIST / Fashion-MNIST) the flatten size is 64*7*7 = 3136.
"""
import torch
import torch.nn as nn


class TwoConvNet(nn.Module):
    def __init__(self, in_channels=3, image_size=32, num_classes=10, hidden=512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        side = image_size // 4
        self.flat_dim = 64 * side * side
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_classes),
            nn.ReLU(),  # output ReLU: outputs are non-negative, compared to one-hot targets
        )

    def forward(self, x):
        return self.classifier(self.features(x))


DATASETS = {
    # name: (in_channels, image_size, mean, std)
    "cifar10": (3, 32, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "fashionmnist": (1, 28, (0.2860,), (0.3530,)),
    "mnist": (1, 28, (0.1307,), (0.3081,)),
}


def build_model(dataset="cifar10"):
    c, s, _, _ = DATASETS[dataset]
    return TwoConvNet(in_channels=c, image_size=s)
