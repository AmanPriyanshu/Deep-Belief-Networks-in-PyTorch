import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import os

def image_beautifier(names, final_name):

	image_names = sorted(names)
	images = [Image.open(x) for x in names]
	widths, heights = zip(*(i.size for i in images))
	total_width = sum(widths)
	max_height = max(heights)
	new_im = Image.new('RGB', (total_width, max_height))

	x_offset = 0
	for im in images:
		new_im.paste(im, (x_offset,0))
		x_offset += im.size[0]

	new_im.save(final_name)
	img = cv2.imread(final_name)
	img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
	cv2.imwrite(final_name, img)

if __name__ == '__main__':
	files = ['DBN_without_pretraining_classifier.csv', 'DBN_with_pretraining_and_input_binarization_classifier.csv', 'DBN_with_pretraining_classifier.csv', 'RBM_pretrained_classifier.csv', 'RBM_without_pretraining_classifier.csv']
	for file in files:
		for feature in [['test loss', 'train loss'], ['test acc', 'train acc']]:
			df = pd.read_csv(file, usecols=feature)
			df = df.values
			name = feature[0][len('test '):]
			plt.cla()
			plt.plot(np.array(range(1, df.shape[0]+1)), df.T[0], label='test - '+name)
			plt.plot(np.array(range(1, df.shape[0]+1)), df.T[1], label='train - '+name)
			plt.legend()
			plt.title(file[:-4]+'_'+name)
			if name=='acc':
				plt.ylim([-0.01, 1.01])
			plt.savefig('./images/'+file[:-4]+'_'+name+'.jpg')

	files = ['./images/'+i for i in sorted(os.listdir('./images/'))]
	DBN = files[:-4]
	RBM = files[-4:]

	DBN_acc = [DBN[i] for i in range(0, len(DBN), 2)]
	DBN_loss = [DBN[i] for i in range(1, len(DBN), 2)]

	RBM_acc = [RBM[i] for i in range(0, len(RBM), 2)]
	RBM_loss = [RBM[i] for i in range(1, len(RBM), 2)]

	image_beautifier(DBN_acc, './images/DBN_acc.jpg')
	image_beautifier(DBN_loss, './images/DBN_loss.jpg')
	image_beautifier(RBM_acc, './images/RBM_acc.jpg')
	image_beautifier(RBM_loss, './images/RBM_loss.jpg')