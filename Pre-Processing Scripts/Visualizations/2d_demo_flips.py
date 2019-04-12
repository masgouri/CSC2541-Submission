import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys


def add_plot(pl, d, r):
	ax = pl.subplot(gs[r, 0]) # row 0, col 0
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	pl.imshow(d)

	ax = pl.subplot(gs[r, 1]) # row 0, col 0
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	pl.imshow(np.fliplr(d))


dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_1_ds8_uni_split.h5', 'r')
data = dataset.get('train_data_im')[:5]

d = data
d2 = np.fliplr(d)

print d.shape
print d2.shape

# Create 2x2 sub plots
gs = gridspec.GridSpec(5, 2)

pl.figure()
add_plot(pl,data[0],0)
add_plot(pl,data[1],1)
add_plot(pl,data[2],2)
add_plot(pl,data[3],3)
add_plot(pl,data[4],4)

#plt.show()
plt.savefig("2d_demo_flip.png")
