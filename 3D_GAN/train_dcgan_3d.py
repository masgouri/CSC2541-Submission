from dcgan_3d import GAN
import argparse
import numpy as np
import h5py

def get_args():
    parser = argparse.ArgumentParser(description='Training 3DGAN:')
    parser.add_argument('-m', '--model', type=str, help='3D_GAN or other types of GANs', default='3D_GAN')
    parser.add_argument('-p', '--data_path', type=str, help='path to training dataset')
    parser.add_argument('--input_shape', type=list, help='input shape of the training data', default=(64,64,64,1))
    parser.add_argument('--output_shape', type=tuple, help='output shape of training data (image)', default=(64,64,64,1))
    parser.add_argument('--epochs', type=int, help='number of epochs for training', default=100)
    parser.add_argument('--batch_size', type=int, help='batch size for training', default = 32)
    parser.add_argument('--save_int', type=int, help='interval to save data at', default=10)
    parser.add_argument('-o', '--out_file', type=str, help='file to save model (e.g. model.h5)')
    parser.add_argument('-c', '--train_class', type=int, help='head class to train', default=0)
    return parser.parse_args()


def train_model(model, data_path, epochs, batch_size, save, out_file, input_shape, output_shape, train_class):
    print("Reading the data")

    data = h5py.File(data_path, 'r')
    x_data = data.get('data_im')[()]
    target = data.get('target')[()]
    l = x_data.shape[0]
    # Get the class we want to train
    index = []
    if train_class == 0:
        for i in range(l):
            if target[i][0] <= 6:
                index.append(i)
        x_data = x_data[index]
    if train_class == 1:
        for i in range(l):
            if target[i][0] == 7:
                index.append(i)
        x_data = x_data[index]
    if train_class == 2:
        for i in range(l):
            if target[i][0] == 8:
                index.append(i)
        x_data = x_data[index]
    
    print('We are working with class ' +  str(train_class) + ' which contains ' + str(l) + ' images')
    
    if (model == '3D_GAN'):
        print("Building the GAN")
        basic_model = GAN(learning_rate_gen=0.0002, learning_rate_disc=0.00005, in_shape=input_shape, out_shape=output_shape)
        print("Built the GAN, now start training")
        basic_model.train(x_data, epochs, save, batch_size, out_file)
    elif:
        pass

def main():
    args =  get_args()
    print("Start training " + args.model)
    train_model(args.model, args.data_path, args.epochs, args.batch_size, args.save_int, args.out_file, args.input_shape, args.output_shape, args.train_class)

if __name__ == '__main__':
    main()
