import pickle
import torch
import imageio

def save_models(D,G, train_hist, train_epoch):
    torch.save(G.state_dict(), PATH+"/generator_param_"+str(train_epoch)+".pkl")
    torch.save(D.state_dict(), PATH +"/discriminator_param_"+str(train_epoch)+".pkl")

    with open(PATH+'/train_hist.pkl', 'wb') as f:
        pickle.dump(train_hist, f)

def save_gif():
    images = []
    for e in range(train_epoch):
        img_name = PATH+'/Fixed_results/MNIST_DCGAN_' + str(e + 1) + '.png'
        images.append(imageio.imread(img_name))
    imageio.mimsave(PATH+'/generation_animation.gif', images, fps=5)