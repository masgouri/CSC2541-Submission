from wgan_3d import WGAN
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


def train_model(data, epochs, batch_size, save, out_file, input_shape, output_shape):
     wgan = WGAN()
     wgan.train(data, epochs=40000, batch_size=128, sample_interval=100)

     return    

def main():
    args = get_args()
    # TODO: set the datapath as an argument
    data = h5py.File('/storage/general/data.h5', 'r')
    x_data = data.get('data_im')[()]
    target = data.get('target')[()]
    l = x_data.shape[0]
    index = []
    # TODO: set the class label as an argument - currently only looking at class 0
    for i in range(l):
        if target[i][0] <= 6:
            index.append(i)
    x_data = x_data[index]
	
    train_model(x_data, args.epochs, args.batch_size, args.save_int, args.out_file, x_data.shape, x_data.shape)


if __name__ == '__main__':
    main()
