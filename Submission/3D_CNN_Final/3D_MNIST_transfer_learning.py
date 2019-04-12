# Restrict to one GPU in multi-GPU systems
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from collections import Counter
import h5py
from itertools import cycle, product
from keras import applications
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from resnet3d import Resnet3DBuilder, basic_block # https://github.com/JihongJu/keras-resnet3d
from scipy import interp
from shutil import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import time


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
    plt.figure(figsize=(20, 16))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=32)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=14)
    plt.yticks(tick_marks, classes, size=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    # Labeling the plot
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.grid(None)
    plt.ylabel('True label', size=18)
    plt.xlabel('Predicted label', size=18)
    plt.tight_layout()
    plt.savefig(output_dir + '/3D_MNIST_transfer_learning_Confusion_Matrix.png')

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('3D MNIST Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('3D MNIST Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.savefig(output_dir + '/3D_MNIST_transfer_learning_Acc_Loss.png')

def create_train_test(datapath, test_size=0.3):
    with h5py.File(datapath, 'r') as dataset:
        data = dataset.get('data_im')[()]
        target = dataset.get('target')[()]
        for i in range(len(target)):
            if target[i][0] <= 6:
                target[i][0] = 0
            if target[i][0] == 7:
                target[i][0] = 1
            if target[i][0] == 8:
                target[i][0] = 2
        X_train, X_test, y_train, y_test = train_test_split(data, target,
                                                            test_size=test_size)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                                  X_train.shape[2], X_train.shape[3], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1],
                                X_test.shape[2], X_test.shape[3], 1)
        y_train = to_categorical(y_train, 3)
        y_test = to_categorical(y_test, 3)
        return X_train, X_test, y_train, y_test

# Setup the output directory and copy this same script to the directory for future reference
output_dir = "output/" + str(time.strftime("%m-%d-%H:%M"))
os.mkdir(output_dir)
copy(__file__, output_dir)

# Download the data from: /hpf/largeprojects/ccm/devin/plastics-data/???
with h5py.File("data/3d-mnist/full_dataset_vectors.h5", "r") as hf:
    X_train = hf["X_train"][:]
    Y_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    Y_test = hf["y_test"][:]

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

## convert to 1 + 4D space (1st argument represents number of rows in the dataset)
X_train = X_train.reshape(X_train.shape[0], 16, 16, 16, 1)
X_test = X_test.reshape(X_test.shape[0], 16, 16, 16, 1)
Y_train = to_categorical(Y_train, 10)
Y_test = to_categorical(Y_test, 10)

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

# model = Resnet3DBuilder.build_resnet_18((16, 16, 16, 1), 10)
model = Resnet3DBuilder.build((16, 16, 16, 1), 10, basic_block, [1, 1, 1, 1], reg_factor=1e-4)
adam = Adam(lr=0.0001)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
earlystop = EarlyStopping(monitor = 'val_acc',
                          min_delta = 0.001,
                          patience = 5,
                          verbose = 2,
                          restore_best_weights = True)
callback_list = [earlystop]

model_info = model.fit(X_train, Y_train,
                       epochs = 100,
                       batch_size = 1024,
                       validation_split = 0.20,
                       callbacks = callback_list)

preds = model.evaluate(X_test, Y_test)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

plot_model_history(model_info)

# model.summary()

Y_hat_test = model.predict(X_test, verbose=1)
cm = confusion_matrix(Y_test.argmax(axis=1), Y_hat_test.argmax(axis=1))
print("Data Distribution for Test Set: ")
te_unique, te_counts = np.unique(Y_test, return_counts=True)
print(dict(zip(te_unique, te_counts)))
print("Confusion Matrix for Test Set: ")
print(cm)
print("")
plot_confusion_matrix(cm, classes=['0', '1', '2'], title='Confusion Matrix for 3D MNIST', normalize=True)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
lw = 2
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_hat_test[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_hat_test.ravel())
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
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
                   ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.05])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for 3D MNIST')
plt.legend(loc="lower right")

plt.savefig(output_dir + '/3D_MNIST_transfer_learning_AUC.png')


model.save(output_dir + '/model.h5')
