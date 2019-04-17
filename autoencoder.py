import torch
import torch.nn as nn
import time

from generator import Generator
from discriminator import Discriminator

MODEL_DIR = "./MNIST_AE_results/"

class Encoder(nn.Module):
    def __init__(self, d=128, latentdim=100):
        super(Encoder, self).__init__()
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
            nn.Conv2d(d*8, self.latentdim, 4, 1, 0, bias=False)
        )    

    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self, model_path: str):
        weight = torch.load(MODEL_DIR + "encoder_param.pkl")
        self.load_state_dict(weight)

class Decoder(nn.Module):
    def __init__(self, d=128, latentdim=100):
        super(Decoder, self).__init__()
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
            nn.ConvTranspose2d(d, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self, model_path: str):
        weight = torch.load(MODEL_DIR + "decoder_param.pkl")
        self.load_state_dict(weight)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.mse = nn.MSELoss()

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
            for inputs in trainloader:
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
        print("Saving model...")
        self.save(MODEL_DIR)
        print('Autoencoder trained.')

    def save(self, model_path: str):
        torch.save(self.encoder.state_dict(), model_path+"encoder_param.pkl")
        torch.save(self.decoder.state_dict(), model_path+"decoder_param.pkl")

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)



    
