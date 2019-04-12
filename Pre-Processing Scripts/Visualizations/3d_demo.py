import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys


filename1 = "/hpf/largeprojects/ccm/devin/plastics-data/general/data.h5"
filename2 = "/hpf/largeprojects/ccm/devin/plastics-data/general/data_ds2.h5"
filename3 = "/hpf/largeprojects/ccm/devin/plastics-data/general/data_ds4.h5"

def blockwise_average_3D(A, S):
	# A is the 3D input array
	# S is the blocksize on which averaging is to be performed

	S = (S,S,S)

	m,n,r = np.array(A.shape)//S
	return A.reshape(m,S[0],n,S[1],r,S[2]).mean((1,3,5))


def load_data(filename):
	dataset = h5py.File(filename, 'r')
	data = dataset.get('data_im')[()]
	target = dataset.get('target')[()]

	idx0 = np.where(target<=6)[0][51]
	idx1 = np.where(target==7)[0][0]
	idx2 = np.where(target==8)[0][2]

	data0 = data[idx0]
	data1 = data[idx1]
	data2 = data[idx2]

	return data0, data1, data2


def add_plot(pl, r, d):
	z,x,y = d.nonzero()

	ax = pl.subplot(gs[r, 0])
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	pl.scatter(x,y,c='red')

	ax = pl.subplot(gs[r, 1])
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	pl.scatter(x,-z,c='blue')

	ax = pl.subplot(gs[r, 2])
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	pl.scatter(y,-z,c='green')


d1, d2, d3 = load_data(filename1)

print d1.shape
print d2.shape
print d3.shape

# Create 2x2 sub plots
gs = gridspec.GridSpec(3, 3)#, width_ratios=[4,2,1], height_ratios=[4,2,1])
pl.figure()

add_plot(pl, 0, d1)
add_plot(pl, 1, d2)
add_plot(pl, 2, d3)

#plt.show()
plt.savefig("3d_demo.png")
