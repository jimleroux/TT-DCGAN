import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import ToPILImage
from tlayers.TTConv import TTConv
from tlayers.TTDeconv import TTDeconv

from utils.visualization import append_images

MODEL_DIR = "./MNIST_AE_results/"


class Encoder(nn.Module):
    def __init__(self, d=128, latentdim=100, TT=False):
        super(Encoder, self).__init__()
        self.latentdim = latentdim
        if TT == False:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=d,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=d,
                    out_channels=d*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=d*2,
                    out_channels=d*4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(0.2),
                nn.Conv2d(
                    in_channels=d*4,
                    out_channels=d*8,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d*8),
                nn.ReLU(0.2),
                nn.Conv2d(
                    in_channels=d*8,
                    out_channels=self.latentdim,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False
                )
            )    
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(
                    in_channels=3,
                    out_channels=d,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(0.2),
                TTConv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,2,2,2],
                    out_ch_modes=[4,4,2,2],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(0.2),
                TTConv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,2,2],
                    out_ch_modes=[4,4,4,2],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(0.2),
                TTConv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,4,2],
                    out_ch_modes=[4,4,4,4],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(d*8),
                nn.ReLU(0.2),
                TTConv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,4,4],
                    out_ch_modes=[5,5,2,2],
                    ranks=[8,4,4,4,1],
                    stride=1,
                    padding=0
                )
            )

    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self, model_path: str):
        weight = torch.load(MODEL_DIR + "encoder_param.pkl")
        self.load_state_dict(weight)


class Decoder(nn.Module):
    def __init__(self, d=128, latentdim=100, TT=False):
        super(Decoder, self).__init__()
        self.latentdim = latentdim
        self.d = d
        if TT == False:
            self.layers = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=latentdim,
                    out_channels=d*8,
                    kernel_size=4,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(d*8),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(
                    in_channels=d*8,
                    out_channels=d*4,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(
                    in_channels=d*4,
                    out_channels=d*2,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(
                    in_channels=d*2,
                    out_channels=d,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(
                    in_channels=d,
                    out_channels=3,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.Tanh()
            )
        else:
            self.layers = nn.Sequential(
                TTDeconv(
                    conv_size=[4,4],
                    inp_ch_modes=[5,5,2,2],
                    out_ch_modes=[4,4,4,4],
                    ranks=[8,4,4,4,1],
                    stride=1,
                    padding=0
                ),
                nn.BatchNorm2d(d*8),
                nn.LeakyReLU(0.2),
                TTDeconv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,4,4],
                    out_ch_modes=[4,4,4,2],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(d*4),
                nn.LeakyReLU(0.2),
                TTDeconv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,4,2],
                    out_ch_modes=[4,4,2,2],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),
                nn.BatchNorm2d(d*2),
                nn.LeakyReLU(0.2),
                TTDeconv(
                    conv_size=[4,4],
                    inp_ch_modes=[4,4,2,2],
                    out_ch_modes=[4,2,2,2],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),                
                # nn.ConvTranspose2d(
                #     in_channels=d*2,
                #     out_channels=d,
                #     kernel_size=4,
                #     stride=2,
                #     padding=1,
                #     bias=False
                # ),
                nn.BatchNorm2d(d),
                nn.LeakyReLU(0.2),
                TTDeconv(
                    conv_size=[4,4],
                    inp_ch_modes=[2,2,2,4],
                    out_ch_modes=[1,1,1,3],
                    ranks=[8,4,4,4,1],
                    stride=2,
                    padding=1
                ),                  
                # nn.ConvTranspose2d(
                #     in_channels=d,
                #     out_channels=3,
                #     kernel_size=4,
                #     stride=2,
                #     padding=1,
                #     bias=False
                # ),
                nn.Tanh()
            )
    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self, model_path: str):
        weight = torch.load(MODEL_DIR + "decoder_param.pkl")
        self.load_state_dict(weight)


class Autoencoder(nn.Module):
    def __init__(self, device, d=100, TT=False):
        super(Autoencoder, self).__init__()
        self.TT = TT
        self.encoder = Encoder(d=d, TT=TT)
        self.decoder = Decoder(d=d, TT=TT)
        self.mse = nn.MSELoss()
        self.device = device

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

    def fit(self, trainloader, lr, n_epochs, print_every=1):
        print("Training autoencoer...")
        start_time = time.time()
        _optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            self.train()
            train_loss = 0
            for inputs, _ in trainloader:
                inputs = inputs.to(self.device)
                recons = self.forward(inputs)
                loss = self.mse(recons, inputs)
                loss.backward()
                _optimizer.step()
                _optimizer.zero_grad()
                train_loss += loss.data.cpu().numpy() * inputs.shape[0]

            train_loss = train_loss / len(trainloader.dataset)

            if (epoch + 1) % print_every == 0:
                epoch_time = self._get_time(start_time, time.time())
                print('epoch: {} | Train loss: {:.3f} | time: {}'.format(
                    epoch + 1,
                    train_loss,
                    epoch_time)
                )
                self.plot_reconstruction(trainloader, epoch + 1)

        print("Saving model...")
        self.save(MODEL_DIR)
        print('Autoencoder trained.')

    def save(self, model_path: str):
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        torch.save(self.encoder.state_dict(), model_path+"encoder_param.pkl")
        torch.save(self.decoder.state_dict(), model_path+"decoder_param.pkl")

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def plot_reconstruction(self, trainloader, epoch):
        sample, _ = trainloader.dataset[round(np.random.uniform(0, len(trainloader.dataset)))]
        sample = sample.to(self.device)
        sample = sample.unsqueeze(0)
        recons = self.forward(sample)

        sample = ToPILImage(mode='RGB')(sample.squeeze(0).cpu())
        recons = ToPILImage(mode='RGB')(recons.squeeze(0).cpu())

        # Create directory for images
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        if not os.path.isdir("./images"+"/reconstructions"):
            os.mkdir("./images"+"/reconstructions")

        combined = append_images([sample, recons])
        combined.save('./images/reconstructions/epoch{}.jpg'.format(epoch))
