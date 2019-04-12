from GAN_2D import WGAN
import argparse
import numpy as np
import h5py
import scipy

def get_args():
    parser = argparse.ArgumentParser(description='Training 2DGAN:')
  
    parser.add_argument('--epochs', type=int, help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, help='batch size for training')
    parser.add_argument('--save_int', type=int, help='interval to save data at')
    parser.add_argument('--out_file', type=str, help='file to save model (e.g. model.h5)')

    return parser.parse_args()

def pad_image(data):
        max_size = max(data[0].shape[0], data[0].shape[1])
        smaller_size = min(data[0].shape[0], data[0].shape[1])

        max_size = 224

        pad_size_l = (max_size - smaller_size)//2
        pad_size_r = smaller_size + (max_size - smaller_size)//2 

        print(pad_size_l)
        print(pad_size_r)
        uniform_data = np.zeros((data.shape[0], max_size, max_size, data.shape[-1]))
        uniform_data[:,pad_size_l:pad_size_r,pad_size_l:pad_size_r,:] = data

def load_data(num):
    """
    @param num: number indicating which angle of the 2D images you want
    """
    #dataset = h5py.File('/hpf/largeprojects/ccm/devin/plastics-data/general/create_2d_array/2D_data_color_4_ds8_uni_split_flips.h5'.format(num), 'r')
    dataset = h5py.File('/home/chantal/2D_data_grayscaled_4_ds8_uni.h5', 'r')
    data = dataset.get('data_im')
    labels = dataset.get('target')

    print(data.shape)
    #---------------- Pre-Process Data ----------------#
    syn_labels = [1,2,3,4,5,6]
    plag_labels = [7]
    norm_labels = [8]

    syn_data = []
    plag_data = []
    norm_data = []

    # separates data into classes
    for d,l in zip(data,labels):
        if l in syn_labels:
            syn_data += [d]
        elif l in norm_labels:
            norm_data += [d]
        elif l in plag_labels:
            plag_data += [d]

    # change this to train with synostosis/plagiocephaly/normal
    X_train = np.asarray(norm_data)
    batch = X_train.shape[0]
    channels = X_train.shape[3]
    height =X_train.shape[1]
    width = X_train.shape[2]
    X_train = np.reshape(X_train, (batch, channels, height, width))

    print("X_train shape", X_train.shape)

    return X_train


def train_model(data):
     wgan = WGAN()
     wgan.train(data, epochs=15000, batch_size=128, save=500)
     return    

def main():
    args = get_args()
    x_train = load_data(4)
    train_model(x_train)


if _name_ == '_main_':
    main()