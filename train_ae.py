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
    epochs = args.epochs
    batch = args.batch
    filter_cst = args.filtercst
    is_tensorized = args.tensorized
    data = args.data
    validation = args.validation

    print("### Loading data ###")
    train_loader = load_dataset(data, batch, is_train=True)
    if validation:
        valid_loader = load_dataset(data, batch, is_train=not(validation))
    else:
        valid_loader = None
    print("### Loaded data ###")

    model = Autoencoder(
        device=DEVICE,
        d=filter_cst,
        TT=is_tensorized
    )
    model.to(DEVICE)
    model.fit(train_loader, lr, epochs, validloader=valid_loader)

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
                        default="cifar",
                        help="Load dataset specified. mnist or cifar(default).")
    parser.add_argument("--validation",
                        action="store_true",
                        help="Specify if you want to use validation set.")
    args = parser.parse_args()
    train(args)    

