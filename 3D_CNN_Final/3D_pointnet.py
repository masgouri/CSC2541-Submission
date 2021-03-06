# Restrict to one GPU in multi-GPU systems
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import Counter
import h5py
from keras import applications
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from resnet3d import Resnet3DBuilder, basic_block # https://github.com/JihongJu/keras-resnet3d
from shutil import copy
from sklearn.model_selection import train_test_split
import time
from presentation_plots import plot_confusion_matrix, plot_rocauc, plot_model_history


def create_train_test(datapath, test_size=0.3):
    files = ['converted.h5']
             # 'ply_data_train1.h5']


    for file in files:
        f = h5py.File(datapath + file, 'r')
        data = f['data'][:]
        target = f['label'][:]

        print(data.shape)
        print(target.shape)

        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=test_size)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                                  X_train.shape[2], X_train.shape[3], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],
                                X_test.shape[2], X_test.shape[3], 1)
        y_train = to_categorical(y_train, 40)
        y_test = to_categorical(y_test, 40)

        return X_train, X_test, y_train, y_test

# Setup the output directory and copy this same script to the directory for future reference
output_dir = "output/" + str(time.strftime("%m-%d-%H:%M"))
os.mkdir(output_dir)
copy(__file__, output_dir)

datapath = "data/modelnet/modelnet40_ply_hdf5_2048/"
test_size = 0.25
X_train, X_test, Y_train, Y_test = create_train_test(datapath, test_size)


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

dim = 64
model = Resnet3DBuilder.build_resnet_50((dim, dim, dim, 1), 40)
# model = Resnet3DBuilder.build((128, 128, 128, 1), 3, basic_block, [1, 1, 1, 1], reg_factor=1e-4)

# Uncomment the following block to do transfer learning!
# model.layers[-1].name = "dense_resnet_1" # Rename final dense layer so correct number of output classes is used (3 instead of MNIST 10)
# model.load_weights('/home/carsonmclean/dev/csc2541/csc2541/output/04-03-03:37/model.h5',
#                    by_name = True) # build_resnet_18
# model.load_weights('/home/carsonmclean/dev/csc2541/csc2541/output/04-09-00:25/model.h5',
#                    by_name = True) # [1,1,1,1]
# End of transfer learning

adam = Adam(lr=0.00001)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
earlystop = EarlyStopping(monitor = 'val_acc',
                          min_delta = 0.01,
                          patience = 5,
                          verbose = 2,
                          restore_best_weights = True)
callback_list = [earlystop]

model_info = model.fit(X_train, Y_train,
                       epochs = 50,
                       batch_size = 16,
                       validation_split = 0.25,
                       callbacks = callback_list)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

Y_hat_test = model.predict(X_test, verbose=1)

plot_confusion_matrix(Y_test,
                      Y_hat_test,
                      ['Synostosis', 'Plagiocephaly', 'Normal'],
                      "3D PointNet",
                      output_dir)
plot_model_history(model_info, "3D PointNet", output_dir)
plot_rocauc(Y_test, Y_hat_test, "3D PointNet", output_dir)


model.save(output_dir + '/model.h5')
