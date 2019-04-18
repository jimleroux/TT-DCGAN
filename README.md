# Project for IFT 6760-A: Matrix and tensor factorization techniques for machine learning 
Implementation of a DCGAN on various Datasets, Cifar-10, MNIST and SVHN. The vanilla DCGAN model will be compared to a fully tensorized DCGAN model. All the training of the convolutional and linear layers are made in the tensor-train format. To use the code you only have to clone the repository with the command:

```bash
git clone https://github.com/jimleroux/tensor_factorization.git
```

To train the vanilla GAN, start by pretraining the weights with an autoencoder use:    

```bash
python train_ae.py --filtercst 64 --batch 64
```  

After you can train the GAN model with:

```bash
python train_net.py --filtercst 64 --batch 64
```

You can monitor the training by looking at the images in `MNIST_DCGAN_results` folder which will be created during training.
