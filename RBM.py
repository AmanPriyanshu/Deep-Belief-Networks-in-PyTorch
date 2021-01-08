import numpy as np
import torch
import random
from tqdm import trange

class RBM:

	def __init__(self, n_visible, n_hidden, lr=0.001, epochs=5, mode='bernoulli', batch_size=32, k=3, optimizer='adam', gpu=False, savefile=None, early_stopping_patience=5):
		self.mode = mode # bernoulli or gaussian RBM
		self.n_hidden = n_hidden #  Number of hidden nodes
		self.n_visible = n_visible # Number of visible nodes
		self.lr = lr # Learning rate for the CD algorithm
		self.epochs = epochs # Number of iterations to run the algorithm for
		self.batch_size = batch_size
		self.k = k
		self.optimizer = optimizer
		self.beta_1=0.9
		self.beta_2=0.999
		self.epsilon=1e-07
		self.m = [0, 0, 0]
		self.v = [0, 0, 0]
		self.m_batches = {0:[], 1:[], 2:[]}
		self.v_batches = {0:[], 1:[], 2:[]}
		self.savefile = savefile
		self.early_stopping_patience = early_stopping_patience
		self.stagnation = 0
		self.previous_loss_before_stagnation = 0
		self.progress = []

		if torch.cuda.is_available() and gpu==True:  
			dev = "cuda:0" 
		else:  
			dev = "cpu"  
		self.device = torch.device(dev)

		# Initialize weights and biases
		std = 4 * np.sqrt(6. / (self.n_visible + self.n_hidden))
		self.W = torch.normal(mean=0, std=std, size=(self.n_hidden, self.n_visible))
		self.vb = torch.zeros(size=(1, self.n_visible), dtype=torch.float32)
		self.hb = torch.zeros(size=(1, self.n_hidden), dtype=torch.float32)

		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)
		
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
	
	def adam(self, g, epoch, index):
		self.m[index] = self.beta_1 * self.m[index] + (1 - self.beta_1) * g
		self.v[index] = self.beta_2 * self.v[index] + (1 - self.beta_2) * torch.pow(g, 2)

		m_hat = self.m[index] / (1 - np.power(self.beta_1, epoch)) + (1 - self.beta_1) * g / (1 - np.power(self.beta_1, epoch))
		v_hat = self.v[index] / (1 - np.power(self.beta_2, epoch))
		return m_hat / (torch.sqrt(v_hat) + self.epsilon)

	def update(self, v0, vk, ph0, phk, epoch):
		dW = (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
		dvb = torch.sum((v0 - vk), 0)
		dhb = torch.sum((ph0 - phk), 0)

		if self.optimizer == 'adam':
			dW = self.adam(dW, epoch, 0)
			dvb = self.adam(dvb, epoch, 1)
			dhb = self.adam(dhb, epoch, 2)

		self.W += self.lr * dW
		self.vb += self.lr * dvb
		self.hb += self.lr * dhb

	def train(self, dataset):
		dataset = dataset.to(self.device)
		learning = trange(self.epochs, desc=str('Starting...'))
		for epoch in learning:
			train_loss = 0
			counter = 0
			for batch_start_index in range(0, dataset.shape[0]-self.batch_size, self.batch_size):
				vk = dataset[batch_start_index:batch_start_index+self.batch_size]
				v0 = dataset[batch_start_index:batch_start_index+self.batch_size]
				ph0, _ = self.sample_h(v0)

				for k in range(self.k):
					_, hk = self.sample_h(vk)
					_, vk = self.sample_v(hk)
				phk, _ = self.sample_h(vk)
				self.update(v0, vk, ph0, phk, epoch+1)
				train_loss += torch.mean(torch.abs(v0-vk))
				counter += 1
			
			self.progress.append(train_loss.item()/counter)
			details = {'epoch': epoch+1, 'loss': round(train_loss.item()/counter, 4)}
			learning.set_description(str(details))
			learning.refresh()
			
			if train_loss.item()/counter > self.previous_loss_before_stagnation and epoch>self.early_stopping_patience+1:
				self.stagnation += 1
				if self.stagnation == self.early_stopping_patience-1:
					learning.close()
					print("Not Improving the stopping training loop.")
					break
			else:
				self.previous_loss_before_stagnation = train_loss.item()/counter
				self.stagnation = 0
		learning.close()
		if self.savefile is not None:
			model = {'W':self.W, 'vb':self.vb, 'hb':self.hb}
			torch.save(model, self.savefile)

	def load_rbm(self, savefile):
		loaded = torch.load(savefile)
		self.W = loaded['W']
		self.vb = loaded['vb']
		self.hb = loaded['hb']

		self.W = self.W.to(self.device)
		self.vb = self.vb.to(self.device)
		self.hb = self.hb.to(self.device)



def trial_dataset():
	dataset = []
	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				t.append(0)
			else:
				t.append(1)
		dataset.append(t)

	for _ in range(1000):
		t = []
		for _ in range(10):
			if random.random()>0.75:
				t.append(1)
			else:
				t.append(0)
		dataset.append(t)

	dataset = np.array(dataset, dtype=np.float32)
	np.random.shuffle(dataset)
	dataset = torch.from_numpy(dataset)
	return dataset

if __name__ == '__main__':
	
	dataset = trial_dataset()

	rbm = RBM(10, 100, epochs=50, mode='bernoulli', lr=0.001, optimizer='adam', gpu=True, savefile='save_example.pt', early_stopping_patience=50)
	print("Before Training:", rbm.vb)
	rbm.train(dataset)
	print("After Training:", rbm.vb)
	rbm.load_rbm('save_example.pt')
	print("After Loading:", rbm.vb)