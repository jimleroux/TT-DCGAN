import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch


def show_result(
        G, latent_dim,
        fixed_z_,
        z_,
        num_epoch,
        device,
        show=False,
        save=False,
        path='result.png',
        isFix=False
    ):
    # fixed noise

    G.eval()
    test_images = G(fixed_z_) if isFix else G(z_)
    G.train()

    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(5*5):
        i = k // 5
        j = k % 5
        ax[i, j].cla() 
        im = np.transpose(test_images[k, :].cpu().data.numpy(), (1,2,0))
        ax[i, j].imshow(im/2+0.5)

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    print ("Saved to :",path)
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

def show_train_hist(
        hist,
        show=False,
        save=False,
        path='Train_hist.png'
    ):

    x = range(len(hist['D_losses']))

    y1 = hist['D_losses']
    y2 = hist['G_losses']

    plt.plot(x, y1, label='D_loss')
    plt.plot(x, y2, label='G_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc=4)
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()
