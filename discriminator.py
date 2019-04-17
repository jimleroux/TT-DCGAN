import torch
import torch.nn as nn

from utils.initialization import normal_init
from autoencoder import Encoder

MODEL_DIR = "./MNIST_AE_results/"

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, latentdim=100):
        super(Discriminator, self).__init__()
        self.latentdim = latentdim
        self.encoder = Encoder(d=d, latentdim=latentdim)
        self.linear = nn.Linear(latentdim, 1)
        self.sigmoid = nn.Sigmoid()

    # forward method
    def forward(self, inp):
        # input = input.view(-1,28,28)
        # print (input.shape)
        x = self.encoder(inp)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    def train_step(self, x, generator, optimizer, criterion, device):
        self.zero_grad()
        mini_batch_size = x.shape[0]
        y_real = torch.ones(mini_batch_size).to(device)  # D(real) = 1
        y_fake = torch.zeros(mini_batch_size).to(device) # D(fake) = 0

        D_real_result = self(x).view((-1))
        D_real_loss = criterion(D_real_result, y_real)

        # Calculate loss for generated sample
        z = torch.randn((mini_batch_size, self.latentdim)).to(device)
        G_result = generator(z) # Generator's result

        D_fake_result = self(G_result.detach())
        D_fake_loss = criterion(D_fake_result, y_fake)

        # Calculating total loss
        D_train_loss = D_real_loss + D_fake_loss

        # Propogate loss backwards and return loss
        D_train_loss.backward()
        optimizer.step()
        return D_train_loss.item()
