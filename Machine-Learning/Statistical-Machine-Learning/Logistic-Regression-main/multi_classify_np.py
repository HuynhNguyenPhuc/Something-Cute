"""multi_classify_np.py
This file is for multi-class classification using numpy
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
from data_util import get_iris_data 
from bin_classify_np import LogisticClassifier, add_one, add_poly_feature

def test(classifier1, classifier2, classifier3, test_x, test_y):
    """test
    TODO:[YC2.2] 
    Compute precision, recall and F1-score based on test samples
    :param classifier: the trained classifier
    :param test_x: test samples
    :param test_y: test labels
    """ 
    c_mat = np.zeros((3,3))

    r1 = classifier1.feed_forward(test_x)
    r2 = classifier2.feed_forward(test_x)
    r3 = classifier3.feed_forward(test_x)

    pred_y = np.concatenate([r1, r2, r3], axis = 1)
    pred_y = np.argmax(pred_y, axis = 1).reshape(-1, 1)

    for i in range(test_y.shape[0]):
        c_mat[test_y[i, 0] - 1, pred_y[i, 0]] += 1

    print(c_mat)

def train_loop(num_epoch, bin_classifier, train_x, train_y):
    """train_loop

    The train loop. Since we have to train three separate classifiers in this assignment, putting the train loop in another function to save space is more sensible
    :param num_epoch: number of epoch to train
    :param bin_classifier: our logistic classifier object
    :param train_x: input data
    :param train_y: data label
    """
    for e in range(num_epoch):    
        y_hat = bin_classifier.feed_forward(train_x)
        loss = bin_classifier.compute_loss(train_y, y_hat)
        grad = bin_classifier.get_grad(train_x, train_y, y_hat)
        #bin_classifier.numerical_check(train_x, train_y1, grad)
        bin_classifier.update_weight(grad, learning_rate)
        print("Epoch %d: loss is %.5f" % (e+1, loss))

if __name__ == "__main__":
    np.random.seed(2017)
     
    # Load data from file
    # Make sure that iris.data is in data/
    train_x, train_y, test_x, test_y = get_iris_data() 

    # Add more features to train_x and test_x
    #train_x = add_poly_feature(train_x, 2)
    #test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)
    
    # Create 3 classifiers
    num_feature = train_x.shape[1]
    bin_classifier1 = LogisticClassifier((num_feature, 1))
    bin_classifier2 = LogisticClassifier((num_feature, 1))
    bin_classifier3 = LogisticClassifier((num_feature, 1))

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.0005 
    
    # Train the first classifier 
    train_y1 = np.copy(train_y)
    # TODO:[YC2.1] Change train_y1 so that train_y1 rows belong to the first class will be 1 while rows belong to the other classes = 0
    train_y1 = (train_y1 == 1).astype(int)

    train_loop(num_epoch, bin_classifier1, train_x, train_y1) 


    # Train the second classifier 
    train_y2 = np.copy(train_y) 
    # TODO:[YC2.1] Change train_y2 so that train_y2 rows belong to the second class will be 1 while rows belong to the other classes = 0
    train_y2 = (train_y2 == 2).astype(int)

    train_loop(num_epoch, bin_classifier2, train_x, train_y2) 


    # Train the third classifier
    train_y3 = np.copy(train_y)
    # TODO:[YC2.1] Change train_y3 so that train_y3 rows belong to the third class will be 1 while rows belong to the other classes = 0
    train_y3 = (train_y3 == 3).astype(int)

    train_loop(num_epoch, bin_classifier3, train_x, train_y3) 


    test(bin_classifier1, bin_classifier2, bin_classifier3, test_x, test_y) 
