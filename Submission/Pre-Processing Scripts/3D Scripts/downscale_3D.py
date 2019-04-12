import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys

#d_sclae = 2
d_scale = 4

#Script from: https://stackoverflow.com/questions/37532184/downsize-3d-matrix-by-averaging-in-numpy-or-alike
def blockwise_average_3D(A, S):
	# A is the 3D input array
	# S is the blocksize on which averaging is to be performed

	S = (S,S,S)

	m,n,r = np.array(A.shape)//S
	return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/data.h5', 'r')
data = dataset.get('data_im')[()]
data_si = dataset.get('data_si')[()]
target = dataset.get('target')[()]

resized_data = np.array(map(lambda d: blockwise_average_3D(d,d_scale), data), dtype=bool)

print data.shape
print resized_data.shape

data = { 'data_im': resized_data, 'data_si': data_si, 'target': target}
h5py_file = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/data_ds' + str(d_scale) + '.h5', 'w')
for dd in data:
        h5py_file.create_dataset(dd, data=data[dd])
h5py_file.close()
