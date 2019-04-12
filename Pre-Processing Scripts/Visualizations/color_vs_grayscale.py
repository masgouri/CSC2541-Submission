import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
import sys

def load_data(filename):
	dataset = h5py.File(filename, 'r')
	data = dataset.get('data_im')[()][0]
	return data

def add_plot(pl, r, c, d, is_gray):
	ax = pl.subplot(gs[r, c])
	ax.axes.get_xaxis().set_visible(False)
	ax.axes.get_yaxis().set_visible(False)
	if is_gray:
		pl.imshow(d, cmap='gray')
	else:
		pl.imshow(d)

d = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds4_uni.h5")
d1 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds8_uni.h5")
d2 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds12_uni.h5")
d3 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds16_uni.h5")
d4 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds20_uni.h5")
d5 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds24_uni.h5")


d6 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds4_uni.h5")
d7 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds8_uni.h5")
d8 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds12_uni.h5")
d9 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds16_uni.h5")
d10 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds20_uni.h5")
d11 = load_data("/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_grayscaled_4_ds24_uni.h5")

d6 = np.squeeze(d6)
d7 = np.squeeze(d7)
d8 = np.squeeze(d8)
d9 = np.squeeze(d9)
d10 = np.squeeze(d10)
d11 = np.squeeze(d11)

print d.shape
print d6.shape

# Create 4x1 sub plots
gs = gridspec.GridSpec(6, 2, height_ratios=[6, 5, 4, 3, 2, 1])

pl.figure()

add_plot(pl, 0, 0, d, 0)
add_plot(pl, 1, 0, d1,0)
add_plot(pl, 2, 0, d2, 0)
add_plot(pl, 3, 0, d3, 0)
add_plot(pl, 4, 0, d4, 0)
add_plot(pl, 5, 0, d5, 0)
add_plot(pl, 0, 1, d6, 1)
add_plot(pl, 1, 1, d7, 1)
add_plot(pl, 2, 1, d8, 1)
add_plot(pl, 3, 1, d9, 1)
add_plot(pl, 4, 1, d10, 1)
add_plot(pl, 5, 1, d11, 1)

plt.savefig("2d_color_vs_grayscaled_demo.png")
