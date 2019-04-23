import os
import argparse

import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

CONFIG_PATH = "./models/AE/config_"
GAN_RESULTS_PATH = "./MNIST_DCGAN_results/"
SAVE_PATH = "./results/"

configs = ["not_TT", "full_tt", "all_TT", "tt_conv_only", "tt_deconv_only"]

names = ["Base", "FTT", "ATT", "TTenc", "TTdec"]

def plot_AE():
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    plt.figure()
    plt.title("Valid MSE loss")
    for conf, name in zip(configs, names):
        valid_loss = json.load(open(CONFIG_PATH + conf + "/output.json", 'r'))["valid_losses"]
        plt.plot(valid_loss, label=name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    plt.savefig(SAVE_PATH + "AEs_loss.png")

hist = "/train_hist.pkl"

exp = ["all_TT_ttfc", "not_TT_ttfc", "tt_conv_only_ttfc"]

na = ["ATT-TTfc", "Baseline-TTfc", "TTenc-TTfc"]

def plot_Gloss():
    plt.style.use('ggplot')
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)

    plt.figure()
    plt.title("Generator loss on fixed Discriminator")
    for conf, name in zip(exp, na):
        valid_loss = pickle.load(open(GAN_RESULTS_PATH + conf + hist, 'rb'))["G_fix_losses"]
        plt.plot(valid_loss[1:-5], label=name)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    if not os.path.isdir(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    plt.savefig(SAVE_PATH + "G_loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TDCGAN")
    parser.add_argument(
        '--AE',
        action="store_true",
        help="Plot Autoencoder losses."
    )
    parser.add_argument(
        "--G",
        action="store_true",
        help="Plot generator losses."
    )
    args = parser.parse_args()

    if args.AE:    
        plot_AE()
    if args.G:
        plot_Gloss()