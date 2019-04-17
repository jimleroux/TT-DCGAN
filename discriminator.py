import torch
import torch.nn as nn

from utils.initialization import normal_init


class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128, latentdim=100):
        super(Discriminator, self).__init__()
        self.latentdim = latentdim
        self.layers = nn.Sequential(
            nn.Conv2d(3, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d*2, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(d*4, d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*8),
            nn.ReLU(0.2),
            nn.Conv2d(d*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.weight_init(mean=0., std=0.02)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, inp):
        # input = input.view(-1,28,28)
        # print (input.shape)
        x = self.layers(inp)
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
