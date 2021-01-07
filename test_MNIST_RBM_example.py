import torch
import numpy as np
import pandas as pd
import os
from RBM import RBM
from MNIST_RBM_example import MNIST
import cv2

if __name__ == '__main__':
	mnist = MNIST()
	train_x, train_y, test_x, test_y = mnist.load_dataset()
	vn = train_x.shape[1]
	hn = 2500

	rbm = RBM(vn, hn)
	rbm.load_rbm('mnist_trained_rbm.pt')
	
	for n in range(10):
		x = test_x[np.where(test_y==n)[0][0]]
		x = x.unsqueeze(0)
		hidden_image = []
		gen_image = []
		for k in range(rbm.k):
			_, hk = rbm.sample_h(x)
			_, vk = rbm.sample_v(hk)
			gen_image.append(vk.numpy())
			hidden_image.append(hk.numpy())
		hidden_image = np.array(hidden_image)
		hidden_image = np.mean(hidden_image, axis=0)
		gen_image = np.array(gen_image)
		gen_image = np.mean(gen_image, axis=0)
		image = x.numpy()

		image = mnist.inv_transform_normalizer(image)[0]
		hidden_image = (hidden_image*255)[0]
		gen_image = mnist.inv_transform_normalizer(gen_image)[0]

		image = np.reshape(image, (28, 28))
		hidden_image = np.reshape(hidden_image, (50, 50))
		gen_image = np.reshape(gen_image, (28, 28))

		image = image.astype(np.int)
		hidden_image = hidden_image.astype(np.int)
		gen_image = gen_image.astype(np.int)

		print(image.shape, hidden_image.shape, gen_image.shape)

		cv2.imwrite('./images_RBM/'+str(n)+'_original_image.jpg', image)
		cv2.imwrite('./images_RBM/'+str(n)+'_hidden_image.jpg', hidden_image)
		cv2.imwrite('./images_RBM/'+str(n)+'_reconstructed_image.jpg', gen_image)