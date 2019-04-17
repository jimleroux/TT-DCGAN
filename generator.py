import torch.nn as nn
from utils.initialization import normal_init


class Generator(nn.Module):
    # initializers
    def __init__(self, d=128, latentdim=100):
        super(generator, self).__init__()
        self.latentdim = latentdim
        self.d = d
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latentdim, d*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(d*8),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(d*8, d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(d*4, d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d*2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(d*2, d, 4, 2, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(d, 3, 4, 2, 1, bias=False)   
            nn.Tanh()
        )
        self.weight_init(mean=0., std=0.02)
    
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, inp):
        # x = F.relu(self.deconv1(input))
        x = inp.view(-1,self.latentdim,1,1)
        x = self.layers(x)
        return x

    def train_step(self, discriminator, mini_batch_size, optimizer, criterion, device):
        self.zero_grad()
        # Generate z with random values
        z = torch.randn((mini_batch_size, self.latentdim)).to(device)
        y = torch.ones(mini_batch_size).to(device)     # Attempting to be real

        # Calculate loss for generator
        # Comparing discriminator's prediction with ones (ie, real)
        G_result = self(z)
        D_result = discriminator(G_result).view((-1))
        G_train_loss = BCE_loss(D_result, y)

        # Propogate loss backwards and return loss
        G_train_loss.backward()
        optimizer.step()
        return G_train_loss.item()
