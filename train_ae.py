import argparse

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from utils.dataloader import load_dataset
from modules.autoencoder import Autoencoder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_DIR = "./configs/"

def train(args):
    config = json.load(open(CONFIG_DIR + args.config, 'r'))
    data = args.data
    validation = args.validation

    lr = config["training"]["lr"]
    epochs = config["training"]["epochs"]
    batch = config["training"]["batch_size"]

    if args.epochs:
        epochs = args.epochs
    if args.batch:
        batch = args.batch
    if args.lr:
        lr = args.lr    

    print("### Loading data ###")
    train_loader = load_dataset(data, batch, is_train=True)
    if validation:
        valid_loader = load_dataset(data, batch, is_train=not(validation))
    else:
        valid_loader = None
    print("### Loaded data ###")

    model = Autoencoder(
        device=DEVICE,
        config=config
    )
    model.to(DEVICE)
    model.fit(train_loader, lr, epochs, validloader=valid_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default="mnist",
                        help="Load dataset specified. mnist(default) or cifar.")
    parser.add_argument("--validation",
                        action="store_true",
                        help="Specify if you want to use validation set.")
    parser.add_argument('--config',
                        type=str,
                        default="MNIST/all_TT.json",
                        help="Path to config file, all_TT.json as default.")
    parser.add_argument('--lr',
                        type=int,
                        default=None)
    parser.add_argument('--epochs',
                        type=int,
                        default=None)
    parser.add_argument('--batch',
                        type=int,
                        default=None)
    args = parser.parse_args()
    train(args)    

