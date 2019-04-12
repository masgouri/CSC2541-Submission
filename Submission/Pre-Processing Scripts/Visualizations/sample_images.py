

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import h5py
import sys

i_index = sys.argv[1]

dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_' + str(i_index) + '_ds8_uni_split_flips.h5', 'r')
data = dataset.get('train_data_im')[:625]

print "Loaded"

r = 25
c = 25
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
	for j in range(c):
		axs[i,j].imshow(data[cnt])
		#axs[i,j].imshow(np.fliplr(data[cnt]))
		axs[i,j].axis('off')
		cnt += 1
fig.savefig("sample_" + str(i_index) + "_fl.png", dpi=1000)
plt.close()
