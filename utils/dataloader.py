import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

def load_dataset(name, batch_size, is_train=True):
    loader = None
    if name == "cifar":
        transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('data', train=is_train, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
        )
    elif name == "mnist":
        transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=is_train, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
        )
    else:
        print("The data specified is not supported.")

    return loader        
