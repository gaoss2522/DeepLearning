# encoding: utf-8
import numpy as np
import h5py
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.resetwarnings()


class DealData:
    def __init__(self):
        self.train_dir = 'datasets/train_catvnoncat.h5'
        self.test_dir = 'datasets/test_catvnoncat.h5'
        self.index = None

    def load_dataset_h5(self):
        # train_data
        print '-' * 30, 'train_data', '-' * 30
        train_data = h5py.File(self.train_dir)
        print 'train_data_contain', np.array(train_data)
        train_set_x = np.array(train_data['train_set_x'])
        train_set_y = np.array(train_data['train_set_y'])
        train_list_classes = np.array(train_data['list_classes'])
        print ('train_set_x.shape', train_set_x.shape)
        print ('train_set_y.shape', train_set_y.shape)
        print ('train_list_classes.shape', train_list_classes.shape)
        print ('train_list_classes', train_list_classes)

        # test_data
        print '-' * 30, 'test_data', '-' * 30
        test_data = h5py.File(self.test_dir)
        print 'test_data_contain', np.array(test_data)
        test_set_x = np.array(test_data['test_set_x'])
        test_set_y = np.array(test_data['test_set_y'])
        test_list_classes = np.array(test_data['list_classes'])
        print ('test_set_x.shape', test_set_x.shape)
        print ('test_set_y.shape', test_set_y.shape)
        print ('list_classes.shape', test_list_classes.shape)
        print ('list_classes', test_list_classes)

        return train_set_x, train_set_y, test_set_x, test_set_y, train_list_classes

    def change_data_struct(self):
        train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = self.load_dataset_h5()
        print '-' * 100
        train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
        print 'trian_set_x_flatten', train_set_x_flatten.shape

        test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
        print 'test_set_x_flatten', test_set_x_flatten.shape

        return train_set_x_flatten/255., test_set_x_flatten/255., train_set_y, test_set_y

