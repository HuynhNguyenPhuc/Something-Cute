import numpy as np
import pandas as pd

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
    dataIris = pd.read_table('./data/iris.dat', delim_whitespace = True, header = None)
    
    dataIris = dataIris.to_numpy()
    dataIris[:, 4] = dataIris[:, 4]
    
    X = dataIris[:, 0:4]
    y = dataIris[:, 4]

    return (X, y)
