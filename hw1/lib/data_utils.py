import cPickle as pickle
import numpy as np
import os
from scipy.misc import imread
import scipy.io as sio


def load_SVHN_mat(_file_path):
    """ load .mat of SVHN dataset from disk """
    mat_content = sio.loadmat(_file_path)

    train_x = mat_content['train_x'].reshape(45000, 1, 28, 28)
    train_label = mat_content['train_label']
    train_label = np.asarray([int(np.argmax(i)) for i in train_label])

    test_x = mat_content['test_x'].reshape(15000, 1, 28, 28)
    test_label = mat_content['test_label']
    test_label = np.asarray([int(np.argmax(i)) for i in test_label])

    print "shape of train_x", train_x.shape
    print "shape of train_label", train_label.shape
    print "shape of test_x", test_x.shape
    print "shape of test_label", test_label.shape
    
    return train_x, train_label, test_x, test_label

    
def get_SVHN_data(num_training=44000, num_validation=1000, num_test=15000):
    """
    Load the SVHN dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    SVHN_path = 'lib/datasets/SVHN.mat'
    X_train, y_train, X_test, y_test = load_SVHN_mat(SVHN_path)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
