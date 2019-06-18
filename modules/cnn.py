import torch
import torch.nn as nn
from modules.tlayers.TTConv_full import TTConv_full
from modules.tlayers.TTDeconv_full import TTDeconv_full

MODEL_DIR = "./models/AE/"


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.id = config["id"]
        self.layers = self.build(config["encoder"])

    def build(self, config):
        encoder_network = []
        for layer in config:
            if layer['type'] == 'tt_conv':
                encoder_network.append(TTConv_full(**layer['param']))
            elif layer['type'] == 'conv':
                encoder_network.append(nn.Conv2d(**layer['param']))
            elif layer['type'] == 'relu':
                encoder_network.append(nn.ReLU())
            elif layer['type'] == 'leaky_relu':
                encoder_network.append(nn.LeakyReLU(**layer['param']))
            elif layer['type'] == 'tanh':
                encoder_network.append(nn.Tanh())
            elif layer['type'] == 'dropout':
                encoder_network.append(nn.Dropout2d(**layer['param']))
            elif layer['type'] == 'batchnorm':
                encoder_network.append(nn.BatchNorm2d(**layer['param']))
            elif layer['type'] == 'linear':
                encoder_network.append(nn.Linear(**layer['param']))
            else:
                raise ValueError("Unsupported layer type supplied.")
        return nn.Sequential(*encoder_network)

    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self):
        model_path = MODEL_DIR + "config_{}/encoder_param.pkl".format(self.id)
        weight = torch.load(model_path)
        self.load_state_dict(weight)


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.id = config["id"]
        self.layers = self.build(config["decoder"])

    def build(self, config):
        encoder_network = []
        for layer in config:
            if layer['type'] == 'tt_deconv':
                encoder_network.append(TTDeconv_full(**layer['param']))
            elif layer['type'] == 'deconv':
                encoder_network.append(nn.ConvTranspose2d(**layer['param']))
            elif layer['type'] == 'relu':
                encoder_network.append(nn.ReLU())
            elif layer['type'] == 'leaky_relu':
                encoder_network.append(nn.LeakyReLU(**layer['param']))
            elif layer['type'] == 'tanh':
                encoder_network.append(nn.Tanh())
            elif layer['type'] == 'dropout':
                encoder_network.append(nn.Dropout2d(**layer['param']))
            elif layer['type'] == 'batchnorm':
                encoder_network.append(nn.BatchNorm2d(**layer['param']))
            elif layer['type'] == 'linear':
                encoder_network.append(nn.Linear(**layer['param']))
            else:
                raise ValueError("Unsupported layer type supplied.")
        return nn.Sequential(*encoder_network)

    def forward(self, inp):
        output = self.layers(inp)
        return output

    def load(self):
        model_path = MODEL_DIR + "config_{}/decoder_param.pkl".format(self.id)
        weight = torch.load(model_path)
        self.load_state_dict(weight)
