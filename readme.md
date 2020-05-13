# Generative Adversarial Network

The GAN generates new handwritten digits! We train a GAN to generate new handwritten digits after showing it pictures of many real handwritten digits. The basic idea behind GAN is that you have two networks (generator and discriminator) competing against each other. The Generator makes “fake” data to pass to the discriminator. The discriminator sees both generated and real training data and predicts if the data it received is real or fake. The generator is constantly trying to outsmart the discriminator by generating better and better fakes.


## Cost and Accuracy 
The graph represent the `cost` of Genearator and Discriminator for each epoch. 

![graph_cost](/MNIST_GAN_results/MNIST_GAN_train_hist.png)

## Genearted Image

![generated_image](/MNIST_GAN_results/results/MNIST_GAN_99.png)


## Installation

pip:

    pip install tensorflow-gpu==1.5.0

Cuda:

    https://developer.nvidia.com/rdp/cudnn-archive

## How to Use

To using this repo, some things you should to know:

* Compatible with both of CPU and GPU, this code can automatically train on CPU or GPU
* To execute run  `python gan.py`.

## Documentation

You can find the API documentation on the pytorch website:

* https://www.tensorflow.org/api_docs/python/