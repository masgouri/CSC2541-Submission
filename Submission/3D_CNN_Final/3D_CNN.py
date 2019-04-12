from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, Input, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc
from keras.callbacks import TensorBoard
import h5py
import numpy as np
from itertools import cycle
from keras.utils import to_categorical
import argparse
from keras import backend as K
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import interp

# include this to avoid mem leaks issue with tf
K.clear_session()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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
    plt.savefig('/home/dsingh/Documents/HPF_Files/Plastic_GANs/3D_CNN/cf.png')


# Adapted from https://www.kaggle.com/shivamb/3d-convolutions-understanding-use-case

def get_args():
    parser = argparse.ArgumentParser(description='Training Baseline 3D CNN:')

    parser.add_argument('--datapath', type=str, help='path for data',
                        default='/home/dsingh/Documents/HPF_Files/Plastic_GANs/3D_CNN/3d_data/data.h5')
    parser.add_argument('--out_file', type=str, help='file to save model (e.g. model.h5)',
                        default='3D_baseline')
    parser.add_argument('--epochs', type=int, default=10)

    return parser.parse_args()


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


def plot_model(model_history):
    # summarize history for accuracy
    plt.plot(model_history.history['acc'])
    plt.plot(model_history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Baseline_3d_accuracy_' + str(time.time()) + '.png')

    # summarize history for loss
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('Baseline_3d_loss_' + str(time.time()) + '.png')


def build_train_baseline_3D(datapath, test_size, output_path, epochs):
    X_train, X_test, y_train, y_test = create_train_test(datapath, test_size)

    input_layer = Input((X_train.shape[1], X_train.shape[2],
                         X_train.shape[3], 1))

    # convolutional layers
    conv_layer1 = Conv3D(filters=4, kernel_size=(3, 3, 3),
                         activation='relu')(input_layer)
    conv_layer2 = Conv3D(filters=8, kernel_size=(3, 3, 3),
                         activation='relu')(conv_layer1)
    """
    # add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    conv_layer3 = Conv3D(filters=16, kernel_size=(3, 3, 3),
                         activation='relu')(pooling_layer1)
    conv_layer4 = Conv3D(filters=32, kernel_size=(3, 3, 3),
                         activation='relu')(conv_layer3)
    """
    pooling_layer2 = MaxPool3D(pool_size=(2, 2, 2))(conv_layer2)

    # perform batch normalization on the convolution outputs
    pooling_layer2 = BatchNormalization()(pooling_layer2)
    flatten_layer = Flatten()(pooling_layer2)

    dense_layer1 = Dense(units=64, activation='relu')(flatten_layer)
    dense_layer1 = Dropout(0.4)(dense_layer1)
    dense_layer2 = Dense(units=16, activation='relu')(dense_layer1)
    dense_layer2 = Dropout(0.4)(dense_layer2)
    output_layer = Dense(units=3, activation='softmax')(dense_layer2)

    # define the model with input layer and output layer
    model = Model(inputs=input_layer, outputs=output_layer)
    model.summary
    model.compile(loss=categorical_crossentropy, optimizer=Adam(lr=0.1),
                  metrics=['acc'])

    # tensorboard = Tensorboard(log_dir="logs/{}".format(time()))

    history = model.fit(x=X_train, y=y_train, batch_size=1, epochs=5,
                        validation_split=0.3)
    #          validation_split=0.3, callbacks=[tensorboard])

    plot_model(history)

    model.save(output_path + '.h5')

    return model, X_test, y_test


def test_model(model, X_test, y_test):
    y_predict = model.predict(X_test)
    y_predict_non_category = [np.argmax(t) for t in y_predict]
    # print('The confusion matrix of the prediction is, ', confusion_matrix(y_test, y_predict_non_category))
    # print('The accuracy of the prediction is, ', accuracy_score(y_test, y_predict_non_category))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    lw = 2
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_predict[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_predict.ravel())
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
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for 3D Baseline')
    plt.legend(loc="lower right")

    plt.savefig('Baseline_3D_AUC_' + str(time.time()) + '.png')

    y_hat_test = model.predict(X_test, verbose=1)
    cm = confusion_matrix(y_test, y_hat_test)
    print("Data Distribution for Test Set: ")
    te_unique, te_counts = np.unique(y_test, return_counts=True)
    print(dict(zip(te_unique, te_counts)))
    print("Confusion Matrix for Test Set: ")
    print(cm)
    print("")
    plot_confusion_matrix(cm, classes=['Synostosis', 'Plagiocephaly', 'Other'], title='Confusion Matrix')

def main():
    args = get_args()
    model, X_test, y_test = build_train_baseline_3D(args.datapath, 0.3, args.out_file, args.epochs)
    test_model(model, X_test, y_test)


if __name__ == '__main__':
    main()



