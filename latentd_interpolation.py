import argparse

import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from modules.generator import Generator

MODEL_DIR = "./MNIST_DCGAN_results/"
CONFIG_DIR = "./configs/MNIST/"

def load_G(config, path):
    conf = json.load(open(config, 'r'))    
    G = Generator(conf)
    weight = torch.load(path)
    G.load_state_dict(weight)

    return G

def interpolation(G, samples, save_path):
    z1 = torch.randn((1,100))
    z2 = torch.randn((1,100))

    fig, ax = plt.subplots(1, samples, figsize=(15,15))
    for i in range(samples):
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)

    ite = 1/samples
    for k in range(samples):
        z = k*ite*z1 + (1 - k*ite)*z2
        images = G(z)
        ax[k].cla()
        im = np.transpose(images.squeeze(0).cpu().data.numpy(), (1,2,0))
        im = torch.from_numpy(im).squeeze().data.numpy()
        ax[k].imshow(im/2+0.5, cmap="gray")

    plt.savefig(save_path)

def main(args):
    config = CONFIG_DIR + args.config + ".json"
    model_path = MODEL_DIR + args.config + "/generator_param_20.pkl"
    G = load_G(config, model_path)
    interpolation(G, args.K, args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default="all_TT",
        help="Config name"
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default="./results/"
    )
    parser.add_argument(
        '--K',
        type=int,
        default=10
    )
    args = parser.parse_args()

    main(args)

