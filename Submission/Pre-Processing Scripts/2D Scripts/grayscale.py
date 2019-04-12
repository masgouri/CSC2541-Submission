import numpy as np
from skimage.transform import resize	## Downloaded skimage to my own python library
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

filename = sys.argv[1]


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	data = dataset.get('data_im')[()]
	target = dataset.get('target')[()]
	return data, target


def rgb2gray(data):

	full_arr = []
	for batch in range((data.shape[0]//500) + 1):

		if batch == data.shape[0]//500:
			batch_data = data
		else:
			batch_data = data[:500]
		np.delete(data, np.s_[0:500], axis=0)

		#Equation is from: https://pillow.readthedocs.io/en/3.2.x/reference/Image.html#PIL.Image.Image.convert
		grayscaled_data = np.dot(batch_data[...,:3], [0.2989, 0.5870, 0.1140])
		full_arr.append(grayscaled_data)

	grayscaled_data = np.array(full_arr)
	return grayscaled_data


data, target = load_data(filename)
print("Original Shape: ")
print("  Data:         " + str(data.shape))
print("  Target:       " + str(target.shape))
print("")

#grayscaled_data = rgb2gray(data)
data = np.array([np.dot(d[...,:3], [0.2989, 0.5870, 0.1140]) for d in data])
data = data.reshape(data.shape[0], data.shape[1], data.shape[2], 1)
print("Grayscaled Shape: ")
print("  Data:         " + str(data.shape))
print("  Target:       " + str(target.shape))
print("")


data = { 'data_im': data, 'target': target }
h5py_file = h5py.File(filename.replace("color", "grayscaled"), 'w')
for dd in data:
	h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
