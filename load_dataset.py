import torch
import numpy as np
import pandas as pd
import os

class MNIST:
	def __init__(self, DATASET_DIR='./dataset/MNIST/'):
		self.DATASET_DIR = DATASET_DIR

	def fit_normalizer(self, x):
		self.min = np.min(x)
		self.max = np.max(x)

	def transform_normalizer(self, x):
		return (x - self.min)/(self.max - self.min)

	def inv_transform_normalizer(self, x):
		return (x * (self.max - self.min)) + self.min

	def load_dataset(self):
		test = pd.read_csv(self.DATASET_DIR+'test.csv')
		test = test.values
		train = pd.read_csv(self.DATASET_DIR+'train.csv')
		train = train.values
		test_x = test.T[1:].T
		test_y = test.T[0]
		train_x = train.T[1:].T
		train_y = train.T[0]

		train_x, test_x = train_x.astype(np.float32), test_x.astype(np.float32)
		self.fit_normalizer(train_x)
		train_x = self.transform_normalizer(train_x)
		test_x = self.transform_normalizer(test_x)

		train_x, train_y, test_x, test_y = torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(test_x), torch.from_numpy(test_y)	

		return train_x, train_y, test_x, test_y