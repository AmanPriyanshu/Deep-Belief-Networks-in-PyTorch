import torch
import numpy as np
import pandas as pd
import os
from RBM import RBM

def load_dataset(DATASET_DIR='./dataset/MNIST/'):
	test = pd.read_csv(DATASET_DIR+'test.csv')
	test = test.values
	train = pd.read_csv(DATASET_DIR+'train.csv')
	train = train.values
	test_x = test.T[1:].T
	test_y = test.T[0]
	train_x = train.T[1:].T
	train_y = train.T[0]

	train_x, train_y, test_x, test_y = torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(test_x), torch.from_numpy(test_y)	

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	train_x, train_y, test_x, test_y = load_dataset()
	rbm_model = RBM(n_visible = 784, n_hidden = 144, lr = 0.05, epochs = 10, mode='bernoulli')
	k=1
	train_op = rbm_model.update(v, K=k)