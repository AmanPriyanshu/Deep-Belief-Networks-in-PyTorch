import torch
import numpy as np
import pandas as pd
import os
from RBM import RBM
from load_dataset import MNIST

if __name__ == '__main__':
	mnist = MNIST()
	train_x, train_y, test_x, test_y = mnist.load_dataset()
	print('MAE for all 0 selection:', torch.mean(train_x))
	vn = train_x.shape[1]
	hn = 2500

	rbm = RBM(vn, hn, epochs=100, mode='bernoulli', lr=0.0005, k=10, batch_size=128, gpu=True, optimizer='adam', savefile='mnist_trained_rbm.pt', early_stopping_patience=10)
	rbm.train(train_x)