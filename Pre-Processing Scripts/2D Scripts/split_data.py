import h5py
import numpy as np
import math
import sys


#filename = '/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds4_uni.h5'
filename = sys.argv[1]


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	data = dataset.get('data_im')[()]
	target = dataset.get('target')[()]
	return data, target


#Shuffle data
def shuffle_data(data, target):
	np.random.seed(225)
	randIndx = np.arange(len(data))
	np.random.shuffle(randIndx)

	data = data[randIndx]
	target = target[randIndx]
	return data, target


#Split data into train/valid/test with 0.8/0.1/0.1 split
def split_data(data, target):
	tr = int(math.ceil(len(data)*0.8))
	va = int(math.ceil(len(data)*0.9))

	trainData, trainTarget = data[:tr], target[:tr]
	validData, validTarget = data[tr:va], target[tr:va]
	testData, testTarget = data[va:], target[va:]

	return trainData, trainTarget, validData, validTarget, testData, testTarget


Data, Target = load_data(filename)
Data, Target = shuffle_data(Data, Target)
trainData, trainTarget, validData, validTarget, testData, testTarget = split_data(Data, Target)

print("Original Shape:")
print("  Data:         " + str(Data.shape))
print("  Target:       " + str(Target.shape))
print("")
print("New Shape:")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("  Valid Data:   " + str(validData.shape))
print("  Valid Target: " + str(validTarget.shape))
print("  Test Data:    " + str(testData.shape))
print("  Test Target:  " + str(testTarget.shape))

data = { 'train_data_im': trainData, 'train_target': trainTarget, 'valid_data_im': validData, 'valid_target': validTarget, 'test_data_im': testData, 'test_target': testTarget }
h5py_file = h5py.File(filename.replace(".h5", "_split.h5"), 'w')
for dd in data:
        h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
