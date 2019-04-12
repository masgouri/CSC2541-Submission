import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys


#filename = '/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds4_uni_split.h5'
filename1 = sys.argv[1]
filename2 = sys.argv[2]


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	trainData, trainTarget = dataset.get('train_data_im')[()], dataset.get('train_target')[()]
	validData, validTarget = dataset.get('valid_data_im')[()], dataset.get('valid_target')[()]
	testData, testTarget = dataset.get('test_data_im')[()], dataset.get('test_target')[()]
	return trainData, trainTarget, validData, validTarget, testData, testTarget


#Shuffle data
def shuffle_data(data, target):
	np.random.seed(225)
	randIndx = np.arange(len(data))
	np.random.shuffle(randIndx)

	data = data[randIndx]
	target = target[randIndx]
	return data, target


def apply_flips(data):
	flipped_data = np.fliplr(data)
	return flipped_data



trainData1, trainTarget1, validData1, validTarget1, testData1, testTarget1 = load_data(filename1)
trainData2, trainTarget2, validData2, validTarget2, testData2, testTarget2 = load_data(filename2)
print("Original Shape:")
print("  Train Data 1:   " + str(trainData1.shape))
print("  Train Target 1: " + str(trainTarget1.shape))
print("  Train Data 2:   " + str(trainData2.shape))
print("  Train Target 2: " + str(trainTarget2.shape))
print("")

trainData1_flipped = np.array(map(lambda img: apply_flips(img), trainData1))
trainData2_flipped = np.array(map(lambda img: apply_flips(img), trainData2))

trainData1 = np.concatenate((trainData1, trainData2_flipped))
trainTarget1 = np.concatenate((trainTarget1, trainTarget2))
print("New Shape:")
print("  Train Data 1:   " + str(trainData1.shape))
print("  Train Target 1: " + str(trainTarget1.shape))


trainData1, trainTarget1 = shuffle_data(trainData1, trainTarget1)
data = { 'train_data_im': trainData1, 'train_target': trainTarget1, 'valid_data_im': validData1, 'valid_target': validTarget1, 'test_data_im': testData1, 'test_target': testTarget1 }
h5py_file = h5py.File(filename1.replace(".h5", "_flips.h5"), 'w')
for dd in data:
        h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
del trainData2_flipped, trainData1

trainData2 = np.concatenate((trainData2, trainData1_flipped))
trainTarget2 = np.concatenate((trainTarget2, trainTarget1))
print("  Train Data 2:   " + str(trainData2.shape))
print("  Train Target 2: " + str(trainTarget2.shape))


trainData2, trainTarget2 = shuffle_data(trainData2, trainTarget2)
if filename1 != filename2:
	data = { 'train_data_im': trainData2, 'train_target': trainTarget2, 'valid_data_im': validData2, 'valid_target': validTarget2, 'test_data_im': testData2, 'test_target': testTarget2 }
	h5py_file = h5py.File(filename2.replace(".h5", "_flips.h5"), 'w')
	for dd in data:
		h5py_file.create_dataset(dd, data=data[dd])
	h5py_file.close()



# Create 4x1 sub plots
#gs = gridspec.GridSpec(4, 1)

#pl.figure()
#ax = pl.subplot(gs[0])
#pl.imshow(d)
#ax = pl.subplot(gs[1])
#pl.imshow(d2)
#ax = pl.subplot(gs[2])
#pl.imshow(d3)
#ax = pl.subplot(gs[3])
#pl.imshow(d4)
#plt.savefig("2d_demo.png")
