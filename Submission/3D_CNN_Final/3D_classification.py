import os
import logging
import pandas as pd
import numpy as np
import argparse
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


def build_data(datapath):
    data = []
    for filename in os.listdir(datapath):
        # TODO: adapt this to the file format we will generate after
        # training the GAN.
        # Here I assume that the each file is a np array with the last element
        # being the label.
        data.append(np.fromfile(filename), dtype=float)
    dataset = pd.DataFrame(data)
    y = dataset[dataset.columns[-1]]
    X = dataset.drop(labels=dataset.columns[-1], axis=1)

    return X, y


def train_SVM(X, y, max_iter, output_path):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    #                                                   test_size=0.25)
    svm_classifier = SVC(kernel='linear', max_iter=max_iter,
                         class_weight='balanced', verbose=1)
    svm_classifier.fit(X_train, y_train)

    # TODO: plot the evolution of loss
    pickle.dump(svm_classifier, open(output_path, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training SVM Classifier:')

    parser.add_argument('-d', '--datapath',
                        type=str, help='path to load data')
    parser.add_argument('-i', '--max_iter', type=int,
                        help='maximum number of iteration')
    parser.add_argument('-o', '--output_path', type=str,
                        help='path to save model')

    args = parser.parse_args()

    logging.info('Build dataset')
    data = build_data(args.datapath)
    train_SVM(data, args.max_iter, args.output_path)
