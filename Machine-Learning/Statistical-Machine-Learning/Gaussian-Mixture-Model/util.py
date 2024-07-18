from data_util import *

def preprocessing():
    X, y = get_iris_data()
    
    classes = np.unique(y)
    num_classes = len(classes)

    y = y.flatten()

    return X, y, classes, num_classes
