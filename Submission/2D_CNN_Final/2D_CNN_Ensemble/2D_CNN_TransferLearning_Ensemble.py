import pandas as pd
import numpy as np

import itertools
from itertools import cycle
import h5py
import time
import sys
import argparse
import math
import os


# Plotting tools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, log_loss, roc_auc_score, classification_report, roc_curve, confusion_matrix, precision_recall_curve, auc
from sklearn.utils import shuffle, class_weight
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils.fixes import signature
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import binarize
from scipy import stats
from scipy import interp


# Get rid of ALL warning messages
import os
import warnings
import tensorflow as tf
import keras
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.applications import InceptionResNetV2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras.layers import Conv3D, MaxPool3D
from keras.layers import Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model


# include this to avoid mem leaks issue with tf
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 24})
sess = tf.Session(config=config)
keras.backend.set_session(sess)





#loading models
path='/home/pouriamashouri/project'  # Where the individual graphs will be saved

#Paths for source data files
#Angle 1
filename1 = "/home/pouriamashouri/project/data/2D_data_color_0_ds8_uni_split_flips.h5"
model_name1 = "/home/pouriamashouri/project/Models/2D_CNN_colour_0_ds8.h5"

#Angle 2
filename2 = "/home/pouriamashouri/project/data/2D_data_color_1_ds8_uni_split_flips.h5"
model_name2 = "/home/pouriamashouri/project/Models/2D_CNN_colour_1_ds8.h5"

#Angle 3
filename3 = "/home/pouriamashouri/project/data/2D_data_color_2_ds8_uni_split_flips.h5"
model_name3 = "/home/pouriamashouri/project/Models/2D_CNN_colour_2_ds8.h5"

#Angle 4
filename4 = "/home/pouriamashouri/project/data/2D_data_color_3_ds8_uni_split_flips.h5"
model_name4 = "/home/pouriamashouri/project/Models/2D_CNN_colour_3_ds8.h5"

#Angle 5
filename5 = "/home/pouriamashouri/project/data/2D_data_color_4_ds8_uni_split_flips.h5"
model_name5 = "/home/pouriamashouri/project/Models/2D_CNN_colour_4_ds8.h5"


# Plotting the AUROCs for each of the individual models

def test_model(testData, testTarget, m_i, model=None, y_predict=None):
    if (y_predict is None):
        y_predict = model.predict(testData)
        
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(testTarget[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testTarget.ravel(), y_predict.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(3)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(3):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 3

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='Average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ' + str(m_i) + ': 2D ROC Baseline')
    plt.legend(loc="lower right")

    plt.savefig('2D_CNN_Ensemble_AUROC.png')


def plot_confusion_matrix(cm, classes, normalize=True, title='Model 0 Confusion matrix', cmap=plt.cm.Greens, m_i=0):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    # Plot the confusion matrix
    plt.figure(figsize=(20, 14))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    plt.savefig('2D_CNN_Ensemble_CM.png')


#Loading and processing the data for each model
def convert_to_categorical(target):
    target[target <= 6] = 0
    target[target == 7] = 1
    target[target == 8] = 2
    target = to_categorical(target)
    return target


def load_data(filename):
    dataset = h5py.File(filename, 'r')
    trainData, trainTarget = dataset.get('train_data_im')[()], dataset.get('train_target')[()]
    validData, validTarget = dataset.get('valid_data_im')[()], dataset.get('valid_target')[()]
    testData, testTarget = dataset.get('test_data_im')[()], dataset.get('test_target')[()]
    trainTarget = convert_to_categorical(trainTarget)
    validTarget = convert_to_categorical(validTarget)
    testTarget = convert_to_categorical(testTarget)
    return trainData, trainTarget, validData, validTarget, testData, testTarget




def make_prediction(filename, model_name, m_i):
    trainData, trainTarget, validData, validTarget, testData, testTarget = load_data(filename)

    testTarget_i = [np.argmax(t) for t in testTarget]

    # Defining the model input and output shape
    img_width = testData.shape[2]
    img_height = testData.shape[1]

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)
    output_shape = len(testTarget[0])

    #Defining each of the model CNNs, loading each model, and then compiling the model after the load
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)

    #Model
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))


    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model.load_weights(model_name)

    opt = keras.optimizers.Adam(lr=0.00004)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    #Make Predictions
    y_hat_test_all = model.predict(testData, verbose=1)
    return y_hat_test_all, np.array(testTarget_i)



angle1_prediction, angle1_target = make_prediction(filename1, model_name1, 0)
angle2_prediction, angle2_target = make_prediction(filename2, model_name2, 1)
angle3_prediction, angle3_target = make_prediction(filename3, model_name3, 2)
angle4_prediction, angle4_target = make_prediction(filename4, model_name4, 3)
angle5_prediction, angle5_target = make_prediction(filename5, model_name5, 4)


if ((angle1_target==angle2_target).all() and (angle1_target==angle3_target).all() and (angle1_target==angle4_target).all() and (angle1_target==angle5_target).all()):
    print("All targets are the same")
else:
    print("Error, targets dont match. Exiting")
    sys.exit(1)

target = angle1_target


final_prediction = (angle1_prediction + angle2_prediction + angle3_prediction + angle4_prediction + angle5_prediction)/5
final_prediction = final_prediction.reshape(-1,3)
final_prediction_c = np.argmax(final_prediction, axis=1).reshape(-1,1)

confusion_matrix_ensemble = confusion_matrix(target, final_prediction_c)
plot_confusion_matrix(confusion_matrix_ensemble, classes=['Synostosis', 'Plagiocephaly', 'Other'], title='Ensemble Confusion Matrix', m_i="ensemble")
target = to_categorical(target)
test_model(None, target, m_i="ensemble", model=None, y_predict=final_prediction)