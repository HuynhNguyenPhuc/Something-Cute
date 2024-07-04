"""data_util.py
This file provides functions to load data for the assignment
Author: Kien Huynh
"""

import numpy as np
import pandas as pd

def get_eclipse_data(path="data/eclipse-data.npz"):
    """get_eclipse_data
    
    Load the data into 4 numpy arrays: train_x, train_y, test_x, test_y and return them
    :param path: path to the eclipse dataset file
    """
    
    f = np.load(path)
    train_x = f['train_x']
    train_y = f['train_y'] 
    test_x = f['test_x']
    test_y = f['test_y']

    return (train_x, train_y, test_x, test_y)

def get_iris_data(path="./data/iris.dat"):
    """get_iris_data
    
    Load the data into 6 numpy arrays:
    * train_x1
    * train_x2
    * train_x3
    * test_x1
    * test_x2
    * test_x3
    :param path: path to the iris dataset file
    """ 
    dataIris = pd.read_table('data/iris.dat', delim_whitespace = True, header = None)
    dataIris.head()

    dataIris = dataIris.to_numpy()
    dataIris[:, 4] = dataIris[:, 4]
    
    x1 = dataIris[dataIris[:,4]==1, 0:4]
    x2 = dataIris[dataIris[:,4]==2, 0:4]
    x3 = dataIris[dataIris[:,4]==3, 0:4]
    y1 = dataIris[dataIris[:,4]==1, 4]
    y2 = dataIris[dataIris[:,4]==2, 4]
    y3 = dataIris[dataIris[:,4]==3, 4]

    train_x1 = x1[0:40,:]
    test_x1 = x1[40:,:]
    train_x2 = x2[0:40,:]
    test_x2 = x2[40:,:]
    train_x3 = x3[0:40,:]
    test_x3 = x3[40:,:]

    train_y1 = y1[0:40].reshape((40,1))
    test_y1 = y1[40:].reshape((10,1))
    train_y2 = y2[0:40].reshape((40,1))
    test_y2 = y2[40:].reshape((10,1))
    train_y3 = y3[0:40].reshape((40,1))
    test_y3 = y3[40:].reshape((10,1))

    train_x = np.concatenate((train_x1, train_x2, train_x3), axis=0)
    train_y = np.concatenate((train_y1, train_y2, train_y3), axis=0)
    test_x = np.concatenate((test_x1, test_x2, test_x3), axis=0)
    test_y = np.concatenate((test_y1, test_y2, test_y3), axis=0)

    return (train_x, train_y, test_x, test_y)
