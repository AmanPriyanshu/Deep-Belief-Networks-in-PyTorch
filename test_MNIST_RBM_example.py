import torch
import numpy as np
import pandas as pd
import os
from RBM import RBM
from MNIST_RBM_example import MNIST

if __name__ == '__main__':
	mnist = MNIST()
	train_x, train_y, test_x, test_y = mnist.load_dataset()
	vn = train_x.shape[1]
	hn = 2500

	rbm = RBM(vn, hn)
	rbm.load_rbm('mnist_trained_rbm.pt')
	
	for n in range(10):
		x = test_x[np.where(test_y==n)[0][0]]
		print(x)

	#_, hk = self.sample_h(vk)