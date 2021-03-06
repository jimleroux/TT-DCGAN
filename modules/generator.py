import torch
import torch.nn as nn

from modules.cnn import Decoder


class Generator(nn.Module):
    # initializers
    def __init__(self, config):
        super(Generator, self).__init__()
        self.decoder = Decoder(config)

    # forward method
    def forward(self, inp):
        # x = F.relu(self.deconv1(input))
        x = inp.view(-1, 100, 1, 1)
        x = self.decoder(x)
        return x

    def train_step(self, discriminator, mini_batch_size, optimizer, criterion, device):
        self.zero_grad()
        # Generate z with random values
        z = torch.randn((mini_batch_size, 100)).to(device)
        y = torch.ones(mini_batch_size).to(device)     # Attempting to be real

        # Calculate loss for generator
        # Comparing discriminator's prediction with ones (ie, real)
        G_result = self(z)
        D_result = discriminator(G_result).view((-1))
        G_train_loss = criterion(D_result, y)

        # Propogate loss backwards and return loss
        G_train_loss.backward()
        optimizer.step()
        return G_train_loss.item()

    def evaluate(self, discriminator, mini_batch_size, criterion, device):
        self.eval()
        with torch.no_grad():
            # Generate z with random values
            z = torch.randn((mini_batch_size, 100)).to(device)
            y = torch.ones(mini_batch_size).to(device)     # Attempting to be real

            # Calculate loss for generator
            # Comparing discriminator's prediction with ones (ie, real)
            G_result = self(z)
            D_result = discriminator(G_result).view((-1))
            G_train_loss = criterion(D_result, y)
        self.train()
        return G_train_loss
