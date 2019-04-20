import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from dataloader import load_dataset
from autoencoder import Autoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(args):
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    filter_cst = args.filtercst
    is_tensorized = args.tensorized
    data = args.data

    transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    print("### Loading data ###")
<<<<<<< HEAD
    train_loader = load_dataset(data)
=======
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True, transform=transform),
        batch_size=batch_size,
        shuffle=True
    )
>>>>>>> dcab58a436e39d3ea3a34208447d54caf48800e7
    print("### Loaded data ###")

    model = Autoencoder(
        device=DEVICE,
        d=filter_cst,
        TT=is_tensorized
    )
    model.to(DEVICE)
    model.fit(train_loader, lr, num_epochs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=20)
    parser.add_argument('--batch',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        type=int,
                        default=0.0003)
    parser.add_argument('--filtercst',
                        type=int,
                        default=128,
                        help="Multiplicative constant for the number of filters.")
    parser.add_argument('--tensorized',
                        action="store_true",
                        help="Specify if you want the model to be tensorized in a TT")
    parser.add_argument('--data',
                        type=str,
                        default="cifar"
                        help="Load dataset specified. mnist or cifar(default).")
    args = parser.parse_args()
    train(args)    

