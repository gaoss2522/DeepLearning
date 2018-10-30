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


class LR:
    def __init__(self, X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.print_cost = print_cost

        self.w, self.b = self.initialize_with_zeros(X_train.shape[0])

    @staticmethod
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    def initialize_with_zeros(self, dim):
        w = np.zeros((dim, 1))
        b = 0
        assert (w.shape == (dim, 1))
        assert (isinstance(w, float) or isinstance(b, int))
        return w, b

    def propagate(self, w, b, X, Y):
        m = X.shape[1]
        A = self.sigmoid(np.dot(w.T, X)+b)   # z
        cost = -np.sum((Y*np.log(A)+(1-Y)*np.log(1-A)))/m # cost function
        dw = np.dot(X, (A-Y).T)/m
        db = np.sum(A-Y)/m

        cost = np.squeeze(cost)

        assert (dw.shape == w.shape)
        assert (cost.shape == ())

        grads = {"dw": dw, "db": db}
        return grads, cost

    def optimize(self, w, b, X, Y, num_iterations, learning_rate, print_cost=False):
        costs = []
        for i in range(num_iterations):
            grads, cost = self.propagate(w, b, X, Y)

            dw = grads["dw"]
            db = grads["db"]

            w = w - learning_rate*dw
            b = b - learning_rate*db

            if i % 100 == 0:
                costs.append(cost)

            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w, "b": b}
        grads = {"dw": dw, "db": db}

        return params, grads, costs

    def predict(self, w, b, X):

        m = X.shape[1]
        Y_prediction = np.zeros((1, m))
        w = w.reshape(X.shape[0], 1)

        A = self.sigmoid(np.dot(w.T, X)+b)

        for i in range(A.shape[1]):
            Y_prediction = np.around(A)

        assert(Y_prediction.shape == (1, m))

        return Y_prediction

    def model(self):

        w = self.w
        b = self.b
        X_train = self.X_train
        Y_train = self.Y_train
        X_test = self.X_test
        Y_test = self.Y_test
        num_iterations = self.num_iterations
        learning_rate = self.learning_rate
        print_cost = self.print_cost

        # Gradient descent (≈ 1 line of code)
        parameters, grads, costs = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

        w = parameters["w"]
        b = parameters["b"]

        Y_prediction_test = self.predict(w, b, X_test)
        Y_prediction_train = self.predict(w, b, X_train)

        print ('-' * 100)
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

        d = {"costs": costs,
             "Y_prediction_test": Y_prediction_test,
             "Y_prediction_train": Y_prediction_train,
             "w": w,
             "b": b,
             "learning_rate": learning_rate,
             "num_iterations": num_iterations}

        print

        return d


def main():
    get_data = DealData()
    train_set_x_flatten, test_set_x_flatten, train_set_y, test_set_y = get_data.change_data_struct()

    lr_model = LR(train_set_x_flatten, train_set_y, test_set_x_flatten, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
    d = lr_model.model()


if __name__ == '__main__':
    main()
