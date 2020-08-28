from sklearn.datasets import fetch_openml
from input_generator import InputGenerator
import numpy as np

class MnistData(InputGenerator):

    def __init__(self, train_length, nrows):
        self.train_length = train_length 
        self.nrows = nrows 

    def load_data(self):
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        data, target = mnist["data"], mnist["target"]

        return data, target 

    def get_get_train_test_data(self, data, target):
        samples = target.shape[0]
        len_train_labels = len(target)
        ######################################################
        # In this dataset, we set numbers less than 5 to zero  
        # and numbers greater than 5 to one.
        ######################################################
        for i in range(len_train_labels):
            if int(target[i]) > 5:
                target[i] = '1'
            else:
                target[i] = '0'
        
        target = target.reshape(1, samples)
        Y_new = np.eye(self.nrows)[target.astype('int32')]
        Y_new = Y_new.T.reshape(self.nrows, samples)
        X_train, X_test = data[:self.train_length].T, data[self.train_length:].T
        Y_train, Y_test = Y_new[:,:self.train_length], Y_new[:,self.train_length:]

        return X_train, X_test, Y_train, Y_test