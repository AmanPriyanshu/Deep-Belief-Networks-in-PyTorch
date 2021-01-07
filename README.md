# Deep-Belief-Networks-in-PyTorch
The aim of this repository is to create RBMs, EBMs and DBNs in generalized manner, so as to allow modification and variation in model types.

## RBM:

Energy-Based Models are a set of deep learning models which utilize physics concept of energy. They determine dependencies between variables by associating a scalar value, which represents the energy to the complete system.

* It is a probabilistic, unsupervised, generative deep machine learning algorithm.
* It belongs to the energy-based model
* RBM is undirected and has only two layers, Input layer, and hidden layer
* No intralayer connection exists between the visible nodes. 
* All visible nodes are connected to all the hidden nodes

In an RBM, we have a symmetric bipartite graph where no two units within the same group are connected. Multiple RBMs can also be stacked and can be fine-tuned through the process of gradient descent and back-propagation. Such a network is called a Deep Belief Network.

The above project allows one to train an RBM and a DBN in PyTorch on both CPU and GPU. Finally let us take a look at some of the reconstructed images.

### Images - Restricted Boltzmann Machine:

![Image-0](./images_RBM/0.jpg)
![Image-1](./images_RBM/1.jpg)
![Image-2](./images_RBM/2.jpg)
![Image-3](./images_RBM/3.jpg)
![Image-4](./images_RBM/4.jpg)
![Image-5](./images_RBM/5.jpg)
![Image-6](./images_RBM/6.jpg)
![Image-7](./images_RBM/7.jpg)
![Image-8](./images_RBM/8.jpg)
![Image-9](./images_RBM/9.jpg)

