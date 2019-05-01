import argparse
import json
import sys

import numpy as np

sys.path.append("../")
from modules.discriminator import Discriminator
from modules.generator import Generator

CONFIG_DIR = "../configs/"

def num_param(args):
    TT = args.fc_tensorized
    config = json.load(open(CONFIG_DIR + args.config, 'r'))

    D = Discriminator(config, TT)
    G = Generator(config)
    G_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    D_params = sum(p.numel() for p in D.parameters() if p.requires_grad)
    params = G_params + D_params
    print("The model has:{} parameters".format(params))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TDCGAN")
    parser.add_argument(
        '--config',
        type=str,
        default="MNIST/all_TT.json",
        help="Path to config file, all_TT.json as default."
    )
    parser.add_argument(
        "--fc_tensorized",
        action="store_true",
        help="Specify to have the last FC in TT format."
    )
    args = parser.parse_args()

    num_param(args)
