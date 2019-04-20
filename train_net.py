import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

from dataloader import load_dataset
from discriminator import Discriminator
from generator import Generator
from utils.saver import save_gif, save_models
from utils.showresults import show_result, show_train_hist

MODEL_DIR = "./MNIST_AE_results/"

def train(args):
    pre_trained = args.pre_trained
    PATH = args.path_results
    lrD = args.lrD
    lrG = args.lrG
    epochs = args.epochs
    batch_size = args.batch
    latent_dim = args.latentdim
    filter_cst = args.filtercst
    device = args.device
    save_every = args.save_every
    data = args.data
    is_tensorized = args.tensorized 

    # Create directory for results
    if not os.path.isdir(PATH):
        os.mkdir(PATH)
    if not os.path.isdir(PATH+'/Random_results'):
        os.mkdir(PATH+'/Random_results')
    if not os.path.isdir(PATH+'/Fixed_results'):
        os.mkdir(PATH+'/Fixed_results')

    print("### Loading data ###")
    train_loader = load_dataset(data, batch_size)
    print("### Loaded data ###")

    print("### Create models ###")
    D = Discriminator(filter_cst, latent_dim, TT=is_tensorized).to(device)
    G = Generator(filter_cst, latent_dim, TT=is_tensorized).to(device)
    model_parameters = filter(lambda p: p.requires_grad, D.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    model_parameters = filter(lambda p: p.requires_grad, G.parameters())
    params += sum([np.prod(p.size()) for p in model_parameters])
    print("The model has:{} parameters".format(params))
    if pre_trained:
        D.encoder.load(MODEL_DIR)
        G.decoder.load(MODEL_DIR)
    
    G_optimizer = optim.Adam(
        G.parameters(),
        lr=lrG,
        betas=(0.5, 0.999)
    )
    D_optimizer = optim.Adam(
        D.parameters(),
        lr=lrD,
        betas=(0.5, 0.999)
    )
    
    train_hist = {
        'D_losses':[],
        'G_losses':[],
        'G_fix_losses':[]
    }
    
    BCE_loss = nn.BCELoss()
    fixed_z_ = torch.randn((5 * 5, latent_dim)).to(device)    # fixed noise
    for epoch in range(epochs):
        if epoch == 1 or epoch%save_every == 0:
            D_test = copy.deepcopy(D)
        D_losses = []
        G_losses = []
        G_fix_losses = []
        for x, _ in train_loader:
            x = x.to(device)
            D_loss = D.train_step(
                x,
                G,
                D_optimizer,
                BCE_loss,
                device
            )
            G_loss = G.train_step(
                D,
                batch_size,
                G_optimizer,
                BCE_loss,
                device
            )
            # G_fix_loss = G.evaluate(
            #     D_test,
            #     batch_size,
            #     BCE_loss,
            #     device
            # )

            D_losses.append(D_loss)
            G_losses.append(G_loss)
            # G_fix_losses.append(G_fix_loss)
        
        meanDloss = torch.mean(torch.FloatTensor(D_losses))
        meanGloss = torch.mean(torch.FloatTensor(G_losses))
        meanGFloss = torch.mean(torch.FloatTensor(G_fix_losses))
        train_hist['D_losses'].append(meanDloss)
        train_hist['G_losses'].append(meanGloss)
        train_hist['G_fix_losses'].append(meanGFloss)
        print(
            "[{:d}/{:d}]: loss_d: {:.3f}, loss_g: {:.3f}, loss_g_fix: {:.3f}".format(
                epoch + 1,
                epochs,
                meanDloss,
                meanGloss,
                meanGFloss
            )
        )
        p = PATH+'/Random_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        fixed_p = PATH+'/Fixed_results/MNIST_DCGAN_' + str(epoch + 1) + '.png'
        z_ = torch.randn((5*5, latent_dim)).to(device)
        show_result(
            G,
            latent_dim,
            fixed_z_,
            z_,
            device,
            (epoch+1),
            save=True,
            path=p,
            isFix=False
        )
        show_result(
            G,
            latent_dim,
            fixed_z_,
            z_,
            device,
            (epoch+1),
            save=True,
            path=fixed_p,
            isFix=True
        )

    print("Training complete. Saving.")
    save_models(
        D,
        G,
        PATH,
        train_hist,
        epochs
    )
    show_train_hist(
        train_hist,
        save=True,
        path=PATH+'/MNIST_DCGAN_train_hist.png'
    )
    save_gif(PATH, epochs)
    
    return D, G

def main():
    parser = argparse.ArgumentParser(description="TDCGAN")
    parser.add_argument(
        "--lrD",
        type=float,
        default=0.002,
        help="Discriminator learning rate"
    )
    parser.add_argument(
        "--lrG",
        type=float,
        default=0.0002,
        help="Generator learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epoch for the training"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=128,
        help="Batch size"
    )
    parser.add_argument(
        "--latentdim",
        type=int,
        default=100,
        help="Size of the latent dimension"
    )
    parser.add_argument(
        "--filtercst",
        type=int,
        default=128,
        help="Multiplicative constant for the number of filters at each layers"
    )
    parser.add_argument(
        "--path_results",
        type=str,
        default="./MNIST_DCGAN_results",
        help="Path to save the results, aka the images"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Specify the computation device"
    )
    parser.add_argument(
        "--pre_trained",
        action="store_true",
        help="Specify the computation device"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save discriminator every n epochs specified to compute loss."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="cifar",
        help="Load dataset specified. mnist or cifar(default)."
    )
    parser.add_argument(
        "--tensorized",
        action="store_true",
        help="Specify to tensorized the model in a TT format"
    )
    args = parser.parse_args()
    
    D, G = train(args)

if __name__ == "__main__":
    main()
