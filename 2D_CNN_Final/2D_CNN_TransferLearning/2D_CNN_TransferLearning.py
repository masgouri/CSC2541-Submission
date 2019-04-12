import os
# Get rid of ALL warning messages
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.applications.inception_v3 import preprocess_input
from keras.applications import InceptionResNetV2

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from keras.callbacks import TensorBoard
from keras.regularizers import l1
import tensorflow as tf

# Plotting tools
import matplotlib
matplotlib.use('Agg')  # To disable X rendering so it works on terminal only
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import itertools
from itertools import cycle

from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, average_precision_score, recall_score, precision_score, log_loss
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, confusion_matrix, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from scipy import interp

import h5py
import numpy as np
from itertools import cycle
import argparse
import time
import math
from scipy import interp

# include this to avoid mem leaks issue with tf
K.clear_session()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
config = tf.ConfigProto( device_count = {'GPU': 1, 'CPU': 24})
sess = tf.Session(config=config)
keras.backend.set_session(sess)


#Paths for source data files
#Angle 1
filename='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/data/2D_data_color_0_ds8_uni_split.h5'
path='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/Models/2D_0_color_ds8'

#Angle 2
#filename='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/data/2D_data_color_1_ds8_uni_split.h5'
#path='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/Models/2D_1_color_ds8'

#Angle 3
#filename='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/data/2D_data_color_2_ds8_uni_split.h5'
#path='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/Models/2D_2_color_ds8'

#Angle 4
#filename='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/data/2D_data_color_3_ds8_uni_split.h5'
#path='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/Models/2D_3_color_ds8'

#Angle 5
#filename='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/data/2D_data_color_4_ds8_uni_split.h5'
#path='/home/dsingh/Documents/HPF_Files/Plastic_GANs/2D_CNN/Models/2D_4_color_ds8'

#Calculating and Plotting AUROC for each of the predictions along with the micro-average AUROC
def test_model(model, testData, testTarget):
    y_predict = model.predict(testData)
    y_predict_non_category = [np.argmax(t) for t in y_predict]
    # print('The confusion matrix of the prediction is, ', confusion_matrix(y_test, y_predict_non_category))
    # print('The accuracy of the prediction is, ', accuracy_score(y_test, y_predict_non_category))

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
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(3), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for 2D Baseline')
    plt.legend(loc="lower right")

    plt.savefig('Baseline_2D_AUC_' + str(time.time()) + '.png')

#Plotting the corresponding confusion matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Greens):
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
    plt.savefig(path+'/cf.png')

#Grouping outcome labels into 1 of 3 classes (0=Synostosis, 1=Plagiocephaly, 2=Normal)
def convert_to_categorical(target):
    target[target <= 6] = 0
    target[target == 7] = 1
    target[target == 8] = 2
    target = to_categorical(target)
    return target

#Loading the data
def load_data(filename):
    dataset = h5py.File(filename, 'r')
    trainData, trainTarget = dataset.get('train_data_im')[()], dataset.get('train_target')[()]
    validData, validTarget = dataset.get('valid_data_im')[()], dataset.get('valid_target')[()]
    testData, testTarget = dataset.get('test_data_im')[()], dataset.get('test_target')[()]
    trainTarget = convert_to_categorical(trainTarget)
    validTarget = convert_to_categorical(validTarget)
    testTarget = convert_to_categorical(testTarget)
    return trainData, trainTarget, validData, validTarget, testData, testTarget

trainData, trainTarget, validData, validTarget, testData, testTarget = load_data(filename)

#Creating baseline neural network model to be used without transfer learning
def create_model(input_shape, output_shape):
    conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    #conv_base=applications.InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    model = Sequential()
    model.add(conv_base)
    #model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.8))

    #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.6))

    #model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.6))

    #model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    #model.add(Dropout(0.8))

    #model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    #model.add(Dropout(0.8))

    model.add(Flatten())
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))

    return model

trainTarget_i = [np.argmax(t) for t in trainTarget]
validTarget_i = [np.argmax(t) for t in validTarget]
testTarget_i = [np.argmax(t) for t in testTarget]

print("--- Data structures: --- ")
print("  Train Data:   " + str(trainData.shape))
print("  Train Target: " + str(trainTarget.shape))
print("  Valid Data:   " + str(validData.shape))
print("  Valid Target: " + str(validTarget.shape))
print("  Test Data:    " + str(testData.shape))
print("  Test Target:  " + str(testTarget.shape))
print("")

# dimensions of our images.
img_width = trainData.shape[2]
img_height = trainData.shape[1]

nb_train_samples = trainData.shape[0]
nb_validation_samples = validData.shape[0]
#epochs = 1  # need to change these values
#batch_size = 1  # need to change these values

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

output_shape = len(trainTarget[0])

print("--- Size of input & output: --- ")
print("  Input Shape:  " + str(input_shape))
print("  Output Shape: " + str(output_shape))
print("")

model = create_model(input_shape, output_shape)

#opt = keras.optimizers.SGD(lr=0.00001)
opt = keras.optimizers.Adam(lr=0.00004)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

class_weights = class_weight.compute_class_weight('balanced', np.unique(trainTarget_i), trainTarget_i)
early_stopping_monitor = EarlyStopping(patience=10)

history=model.fit(
    x=trainData,
    y=trainTarget,
    batch_size=50,
    epochs=500,
    validation_data=(validData, validTarget),
    class_weight=class_weights,
    callbacks=[early_stopping_monitor],
    verbose=1
)

#Save the model
model.save_weights(path+'/2D_CNN_colour_ds8.h5')

#Evaluate the Model
results = model.evaluate(testData, testTarget)
predictions = model.predict_classes(testData)
print(model.metrics_names)
print(results)

# Create Accuracy and Loss Over Time
history_dict = history.history
history_dict.keys()

# dict_keys(['loss', 'acc', 'val_acc', 'val_loss'])
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize = (10,8))
plt.style.use('fivethirtyeight')
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig(path+'/NNloss.png')

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(path+'/NNAccuracy.png')

probs = model.predict_proba(testData) #[:, 1]

#calculations for confusion matrix
y_hat_train = model.predict_classes(trainData, verbose=1)
confusion_matrix_train = confusion_matrix(trainTarget_i, y_hat_train)
print("Data Distribution for Training Set: ")
tr_unique, tr_counts = np.unique(trainTarget_i, return_counts=True)
print(dict(zip(tr_unique, tr_counts)))
print("Confusion Matrix for Training Set: ")
print(confusion_matrix_train)
print("")

y_hat_valid = model.predict_classes(validData, verbose=1)
confusion_matrix_valid = confusion_matrix(validTarget_i, y_hat_valid)
print("Data Distribution for Validation Set: ")
va_unique, va_counts = np.unique(validTarget_i, return_counts=True)
print(dict(zip(va_unique, va_counts)))
print("Confusion Matrix for Validation Set: ")
print(confusion_matrix_valid)
print("")

y_hat_test = model.predict_classes(testData, verbose=1)
cm = confusion_matrix(testTarget_i, y_hat_test)
print("Data Distribution for Test Set: ")
te_unique, te_counts = np.unique(testTarget_i, return_counts=True)
print(dict(zip(te_unique, te_counts)))
print("Confusion Matrix for Test Set: ")
print(cm)
print("")
plot_confusion_matrix(cm, classes=['Synostosis', 'Plagiocephaly', 'Other'], title='Confusion Matrix')

test_model(model, testData, testTarget)