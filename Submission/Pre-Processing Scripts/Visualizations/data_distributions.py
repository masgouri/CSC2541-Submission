import numpy as np
import h5py


dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_1_ds24_uni.h5', 'r')
data = dataset.get('data_im')[()]
target = dataset.get('target')[()]

print("Data Distribution for 2D Set: ")
te_unique, te_counts = np.unique(target, return_counts=True)
print(dict(zip(te_unique, te_counts)))


dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/data_ds4.h5', 'r')
data = dataset.get('data_im')[()]
target = dataset.get('target')[()]

print("Data Distribution for 3D Set: ")
te_unique, te_counts = np.unique(target, return_counts=True)
print(dict(zip(te_unique, te_counts)))
