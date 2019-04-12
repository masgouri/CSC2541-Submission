import numpy as np
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

filename = sys.argv[1]
#d_scale = sys.argv[2]


def load_data(filename):
        dataset = h5py.File(filename, 'r')
        data = dataset.get('data_im')[()]
        target = dataset.get('target')[()]
        return data, target


def rescale_image(data):
	rescaled_image = data*255.
	if np.amax(rescaled_image) <= 1:
		return rescaled_image
	else:
		return data


data, target = load_data(filename)
print("Original Shape: ")
print("  Data:         " + str(data.shape))
print("  Target:       " + str(target.shape))
print("")

image_rescaled = np.array(map(lambda img: rescale_image(img), data))
print("Rescaled Shape: ")
print("  Data:         " + str(image_rescaled.shape))
print("  Target:       " + str(target.shape))
print("")

data = { 'data_im': image_rescaled, 'target': target }
h5py_file = h5py.File(filename, 'w')
#h5py_file = h5py.File(filename.replace(".h5", "_t.h5"), 'w')
for dd in data:
        h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
