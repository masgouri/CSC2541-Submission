import numpy as np
from skimage.transform import resize	## Downloaded skimage to my own python library
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

filename = sys.argv[1]
d_scale = sys.argv[2]


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	data = dataset.get('data_im')[()]
	target = dataset.get('target')[()]
	return data, target


def resize_image(data):
	image_resized = map(lambda img: resize(img, (img.shape[0] / int(d_scale), img.shape[1] / int(d_scale)), anti_aliasing=True), data)
	image_resized = np.array(image_resized)
	image_resized = image_resized/255.
	return image_resized


def pad_image(data):
	max_size = max(data[0].shape[0], data[0].shape[1])
	smaller_size = min(data[0].shape[0], data[0].shape[1])

	pad_size_l = (max_size - smaller_size)//2
	pad_size_r = smaller_size + (max_size - smaller_size)//2

	uniform_data = np.zeros((data.shape[0], max_size, max_size, data.shape[-1]))
	uniform_data[:,:,pad_size_l:pad_size_r,:] = data
	return uniform_data


data, target = load_data(filename)
print("Original Shape: ")
print("  Data:         " + str(data.shape))
print("  Target:       " + str(target.shape))
print("")

image_resized = resize_image(data)
print("Downscaled & Normalized Shape: ")
print("  Data:         " + str(image_resized.shape))
print("  Target:       " + str(target.shape))
print("")

del data

uniform_data = pad_image(image_resized)
print("Uniform Shape: ")
print("  Data:         " + str(uniform_data.shape))
print("  Target:       " + str(target.shape))


data = { 'data_im': uniform_data, 'target': target }
h5py_file = h5py.File(filename.replace(".h5", "_ds" + d_scale + "_uni.h5"), 'w')
for dd in data:
	h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
