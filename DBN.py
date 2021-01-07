import numpy as np
import torch
import random
from tqdm import trange
from RBM import RBM

class DBN:
	def __init__(self, layers, gpu=False):
		self.layers = layers

		if torch.cuda.is_available() and gpu==True:  
			dev = "cuda:0" 
		else:  
			dev = "cpu"  
		self.device = torch.device(dev)

	def train_hidden(self):
		pass