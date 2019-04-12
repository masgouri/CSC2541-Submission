import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys


#filename = '/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds4_uni_split.h5'
filename = sys.argv[1]


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


def apply_flips_and_rotations(data, target):
	#d2 = np.fliplr(d) #Not needed anymore, np.rot(90, k=2) does the same thing
	data_rot1 = np.array(map(lambda arr: np.rot90(arr, k=1), data))
	data_rot2 = np.array(map(lambda arr: np.rot90(arr, k=2), data))
	data_rot3 = np.array(map(lambda arr: np.rot90(arr, k=3), data))

	data = np.concatenate((data, data_rot1, data_rot2, data_rot3))
	target = np.concatenate((target, target, target, target))
	return data, target



trainData, trainTarget, validData, validTarget, testData, testTarget = load_data(filename)
print("Original Shape:")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("")

trainData, trainTarget = apply_flips_and_rotations(trainData, trainTarget)
trainData, trainTarget = shuffle_data(trainData, trainTarget)

print("New Shape:")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("  Valid Data:   " + str(validData.shape))
print("  Valid Target: " + str(validTarget.shape))
print("  Test Data:    " + str(testData.shape))
print("  Test Target:  " + str(testTarget.shape))

data = { 'train_data_im': trainData, 'train_target': trainTarget, 'valid_data_im': validData, 'valid_target': validTarget, 'test_data_im': testData, 'test_target': testTarget }
h5py_file = h5py.File(filename.replace(".h5", "_flips.h5"), 'w')
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
