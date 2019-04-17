import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from autoencoder import Autoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    img_size = 64

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    print("### Loading data ###")
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data',
        train=True,
        download=True,
        transform=transform),
        batch_size=batch_size,
        shuffle=True
    )
    print("### Loaded data ###")

    model = Autoencoder()
    model.fit(train_loader, lr, num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs',
                        type=int,
                        default=20)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        type=int,
                        default=0.0003)
    args = parser.parse_args()
    train(args)    

