"""multi_classify_tf.py
This file is for multi-class classification using TensorFlow
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_util import get_iris_data
from bin_classify_np import add_one, add_poly_feature

if __name__ == "__main__":
    # Random seed is fixed so that every run is the same
    # This makes it easier to debug
    np.random.seed(2017)

    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_iris_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]  

    # Add more features to train_x and test_x
    # train_x = add_poly_feature(train_x, 2)
    # test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)

    # Train the first classifier 
    train_y1 = np.copy(train_y)
    # TODO:[YC2.1] Change train_y1 so that train_y1 rows belong to the first class will be 1 while rows belong to the other classes = 0
    train_y1 = ((train_y1 == 1).astype(int)).reshape(-1, 1)

    # Train the second classifier 
    train_y2 = np.copy(train_y) 
    # TODO:[YC2.1] Change train_y2 so that train_y2 rows belong to the second class will be 1 while rows belong to the other classes = 0
    train_y2 = ((train_y2 == 2).astype(int)).reshape(-1, 1)

    # Train the third classifier
    train_y3 = np.copy(train_y)
    # TODO:[YC2.1] Change train_y3 so that train_y3 rows belong to the third class will be 1 while rows belong to the other classes = 0
    train_y3 = ((train_y3 == 3).astype(int)).reshape(-1, 1)

    # TODO:[YC2.3] Create TF placeholders to feed train_x and train_y when training
    x = tf.constant(train_x, dtype=tf.float32, shape=train_x.shape)
    y1 = tf.constant(train_y1, dtype=tf.float32, shape=train_y1.shape)
    y2 = tf.constant(train_y2, dtype=tf.float32, shape=train_y2.shape)
    y3 = tf.constant(train_y3, dtype=tf.float32, shape=train_y3.shape)

    # TODO:[YC2.3] Create weights (W) using TF variables
    w1 = tf.Variable(tf.random.normal([train_x.shape[1], 1], 0, tf.sqrt(2./ 6.)))
    w2 = tf.Variable(tf.random.normal([train_x.shape[1], 1], 0, tf.sqrt(2./ 6.)))
    w3 = tf.Variable(tf.random.normal([train_x.shape[1], 1], 0, tf.sqrt(2./ 6.)))

    # TODO:[YC2.3] Create a feed-forward operator
    pred1 = None
    pred2 = None
    pred3 = None

    # TODO:[YC2.3] Write the cost function
    cost1 = None
    cost2 = None
    cost3 = None
    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005    

    # TODO:[YC2.3] Implement GD
    optimizer1 = None
    optimizer2 = None
    optimizer3 = None

    epsilon = 1e-6

    # Start training   
    for epoch in range(num_epoch):
        with tf.GradientTape() as tape1:
            pred1 = 1. / (1. + tf.exp(-tf.matmul(x, w1)))
            pred1 = tf.clip_by_value(pred1, epsilon, 1. - epsilon)
            cost1 = tf.reduce_mean(-(y1 * tf.math.log(pred1) + (1 - y1) * tf.math.log(1 - pred1)), axis = 0)

        with tf.GradientTape() as tape2:
            pred2 = 1. / (1. + tf.exp(-tf.matmul(x, w2)))
            pred2 = tf.clip_by_value(pred2, epsilon, 1. - epsilon)
            cost2 = tf.reduce_mean(-(y2 * tf.math.log(pred2) + (1 - y2) * tf.math.log(1 - pred2)), axis = 0)

        with tf.GradientTape() as tape3:
            pred3 = 1. / (1. + tf.exp(-tf.matmul(x, w3)))
            pred3 = tf.clip_by_value(pred3, epsilon, 1. - epsilon)
            cost3 = tf.reduce_mean(-(y3 * tf.math.log(pred3) + (1 - y3) * tf.math.log(1 - pred3)), axis = 0)

        gradient1 = tape1.gradient(cost1, [w1])
        w1.assign_sub(learning_rate * gradient1[0])

        gradient2 = tape2.gradient(cost2, [w2])
        w2.assign_sub(learning_rate * gradient2[0])

        gradient3 = tape3.gradient(cost3, [w3])
        w3.assign_sub(learning_rate * gradient3[0])

        print(f"Epoch {epoch + 1}: loss1 is {cost1.numpy()[0]:.5f}, loss2 is {cost2.numpy()[0]:.5f}, loss3 is {cost3.numpy()[0]:.5f}")

    # TODO:[YC2.3] Compute test result using confusion matrix
    c_mat = np.zeros((3,3))
    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    pred_y1 = (1. / (1. + tf.exp(-tf.matmul(test_x, w1)))).numpy().reshape(-1, 1)
    pred_y2 = (1. / (1. + tf.exp(-tf.matmul(test_x, w2)))).numpy().reshape(-1, 1)
    pred_y3 = (1. / (1. + tf.exp(-tf.matmul(test_x, w3)))).numpy().reshape(-1, 1)

    pred_y = np.argmax(np.concatenate([pred_y1, pred_y2, pred_y3], axis = 1), axis = 1).reshape(-1, 1)
    
    for i in range(test_y.shape[0]):
        c_mat[test_y[i, 0] - 1, pred_y[i, 0]] += 1

    print(c_mat)
