import numpy as np
from skimage.transform import resize	## Downloaded skimage to my own python library
import h5py
import sys
import warnings
warnings.filterwarnings('ignore')

filename = sys.argv[1]


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	trainData, trainTarget = dataset.get('train_data_im')[()], dataset.get('train_target')[()]
	validData, validTarget = dataset.get('valid_data_im')[()], dataset.get('valid_target')[()]
	testData, testTarget = dataset.get('test_data_im')[()], dataset.get('test_target')[()]
	return trainData, trainTarget, validData, validTarget, testData, testTarget

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


trainData, trainTarget, validData, validTarget, testData, testTarget = load_data(filename)
print("Original Shape: ")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("  Valid Data:   " + str(validData.shape))
print("  Valid Target: " + str(validTarget.shape))
print("  Test Data:    " + str(testData.shape))
print("  Test Target:  " + str(testTarget.shape))
print("")

#grayscaled_data = rgb2gray(data)
trainData = np.array([np.dot(d[...,:3], [0.2989, 0.5870, 0.1140]) for d in trainData])
trainData = trainData.reshape(trainData.shape[0], trainData.shape[1], trainData.shape[2], 1)
validData = np.array([np.dot(d[...,:3], [0.2989, 0.5870, 0.1140]) for d in validData])
validData = validData.reshape(validData.shape[0], validData.shape[1], validData.shape[2], 1)
testData = np.array([np.dot(d[...,:3], [0.2989, 0.5870, 0.1140]) for d in testData])
testData = testData.reshape(testData.shape[0], testData.shape[1], testData.shape[2], 1)
print("Grayscaled Shape: ")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("  Valid Data:   " + str(validData.shape))
print("  Valid Target: " + str(validTarget.shape))
print("  Test Data:    " + str(testData.shape))
print("  Test Target:  " + str(testTarget.shape))
print("")


data = { 'train_data_im': trainData, 'train_target': trainTarget, 'valid_data_im': validData, 'valid_target': validTarget, 'test_data_im': testData, 'test_target': testTarget }
h5py_file = h5py.File(filename.replace("color", "grayscaled"), 'w')
for dd in data:
        h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
