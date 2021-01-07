import numpy as np
import torch
import random

class RBM:

	def __init__(self, n_visible, n_hidden, lr=0.001, epochs=5, mode='bernoulli', batch_size=32, k=3):
		self.mode = mode # bernoulli or gaussian RBM
		self.n_hidden = n_hidden #  Number of hidden nodes
		self.n_visible = n_visible # Number of visible nodes
		self.lr = lr # Learning rate for the CD algorithm
		self.epochs = epochs # Number of iterations to run the algorithm for
		self.batch_size = batch_size
		self.k = k

		# Initialize weights and biases
		std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
		self.W = torch.normal(mean=0, std=std, size=(self.n_hidden, self.n_visible))
		self.vb = torch.zeros(size=(1, self.n_visible), dtype=torch.float32)
		self.hb = torch.zeros(size=(1, self.n_hidden), dtype=torch.float32)
		
	def sample_h(self, x):
		wx = torch.mm(x, self.W.t())
		activation = wx + self.hb
		p_h_given_v = torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_h_given_v, torch.bernoulli(p_h_given_v)
		else:
			return p_h_given_v, torch.add(p_h_given_v, torch.normal(mean=0, std=1, size=p_h_given_v.shape))

	def sample_v(self, y):
		wy = torch.mm(y, self.W)
		activation = wy + self.vb
		p_v_given_h =torch.sigmoid(activation)
		if self.mode == 'bernoulli':
			return p_v_given_h, torch.bernoulli(p_v_given_h)
		else:
			return p_v_given_h, torch.add(p_v_given_h, torch.normal(mean=0, std=1, size=p_v_given_h.shape))
	def update(self, v0, vk, ph0, phk):
		self.W += self.lr * (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		self.vb += self.lr * torch.sum((v0 - vk), 0)
		self.hb += self.lr * torch.sum((ph0 - phk), 0)

	def train(self, dataset):
		for epoch in range(self.epochs):
			train_loss = 0
			counter = 0
			for batch_start_index in range(0, dataset.shape[0]-self.batch_size, self.batch_size):
				vk = dataset[batch_start_index:batch_start_index+self.batch_size]
				v0 = dataset[batch_start_index:batch_start_index+self.batch_size]
				ph0, _ = self.sample_h(v0)

				for k in range(self.k):
					_, hk = self.sample_h(vk)
					_, vk = self.sample_v(hk)
					vk[v0<0] = v0[v0<0]
				phk, _ = self.sample_h(vk)
				self.update(v0, vk, ph0, phk)
				train_loss += torch.mean(torch.abs(v0-vk))
				counter += 1
			print('epoch: '+str(epoch)+' loss: '+str(train_loss/counter))

def trial_dataset():
	dataset = []
	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				if random.random() > 0.5:
					t.append(0)
				else:
					t.append(-1)
			else:
				t.append(1)
		dataset.append(t)

	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				if random.random() > 0.5:
					t.append(0)
				else:
					t.append(1)
			else:
				t.append(-1)
		dataset.append(t)

	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				if random.random() > 0.5:
					t.append(1)
				else:
					t.append(-1)
			else:
				t.append(0)
		dataset.append(t)

	dataset = np.array(dataset, dtype=np.float32)
	np.random.shuffle(dataset)
	dataset = torch.from_numpy(dataset)
	return dataset

if __name__ == '__main__':
	
	dataset = trial_dataset()

	rbm = RBM(10, 50, epochs=100, mode='bernoulli', lr=0.1)
	rbm.train(dataset)