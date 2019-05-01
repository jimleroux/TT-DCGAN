import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from modules.cnn import Decoder, Encoder
from utils.showresults import show_loss, show_recons

MODEL_DIR = "./models/"

class Autoencoder(nn.Module):
    def __init__(self, device, config):
        super(Autoencoder, self).__init__()
        self.id = config["id"]
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.mse = nn.MSELoss()
        self.device = device

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

    def fit(
            self,
            trainloader,
            lr,
            n_epochs,
            validloader=None,
            print_every=1
        ):
        print("Training autoencoder...")
        start_time = time.time()
        _optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_losses = []
        valid_losses = []
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
            train_losses.append(train_loss)
            
            valid_loss = 0
            if validloader is not None:
                self.eval()
                with torch.no_grad():
                    for inputs, _ in validloader:
                        inputs = inputs.to(self.device)
                        recons = self.forward(inputs)
                        loss = self.mse(recons, inputs)
                        valid_loss += loss.data.cpu().numpy() * inputs.shape[0]
                    valid_loss = valid_loss / len(validloader.dataset)
                    valid_losses.append(valid_loss)

            if (epoch + 1) % print_every == 0:
                epoch_time = self._get_time(start_time, time.time())
                print('epoch: {} | Train loss: {:.3f} | Valid loss: {:.3f} | time: {}'.format(
                    epoch + 1,
                    train_loss,
                    valid_loss,
                    epoch_time)
                )

                self.plot_reconstruction(trainloader, epoch + 1)
        if validloader is not None:
            show_loss(valid_losses)

        losses = {
            "train_losses": train_losses,
            "valid_losses": valid_losses
        }

        print("Saving model...")
        self.save(MODEL_DIR, losses)
        print('Autoencoder trained.')

    def save(self, model_path: str, losses: dict):
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
        if not os.path.isdir(model_path + "AE"):
            os.mkdir(model_path + "AE/")
        if not os.path.isdir(model_path + "AE/config_{}/".format(self.id)):
            os.mkdir(model_path + "AE/config_{}/".format(self.id))
        model_path = model_path + "AE/config_{}/".format(self.id)
        torch.save(self.encoder.state_dict(), model_path+"encoder_param.pkl")
        torch.save(self.decoder.state_dict(), model_path+"decoder_param.pkl")
        with open(model_path + "output.json", "w") as f:
            json.dump(losses, f)  

    def _get_time(self, starting_time, current_time):
        total_time = current_time - starting_time
        minutes = round(total_time // 60)
        seconds = round(total_time % 60)
        return '{} min., {} sec.'.format(minutes, seconds)

    def plot_reconstruction(self, trainloader, epoch, num_samp=5):
        samples = []
        recons = []
        for _ in range(num_samp):
            sample, _ = trainloader.dataset[round(np.random.uniform(0, len(trainloader.dataset)))]
            sample = sample.to(self.device)
            sample = sample.unsqueeze(0)
            rec = self.forward(sample)

            samples.append(sample)
            recons.append(rec)

        images = [samples, recons]

        # Create directory for images
        if not os.path.isdir("./images"):
            os.mkdir("./images")
        if not os.path.isdir("./images/reconstructions"):
            os.mkdir("./images/reconstructions")
        if not os.path.isdir("./images/reconstructions/config_{}".format(self.id)):
            os.mkdir("./images/reconstructions/config_{}".format(self.id))
        image_path = "./images/reconstructions/config_{}/".format(self.id)
        show_recons(images, epoch, num_samp, image_path + "epoch{}.jpg".format(epoch))
