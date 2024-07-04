"""bin_classify_tf.py
This file is for binary classification using TensorFlow
Author: Kien Huynh
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from data_util import get_eclipse_data
from bin_classify_np import add_one, add_poly_feature

if __name__ == "__main__":
    # Random seed is fixed so that every run is the same
    # This makes it easier to debug
    np.random.seed(2017)

    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_eclipse_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]  

    # Add more features to train_x and test_x
    train_x = add_poly_feature(train_x, 2)
    test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)
   
    # TODO:[YC1.9] Create TF placeholders to feed train_x and train_y when training
    x = tf.constant(train_x, dtype = tf.float32, shape = train_x.shape)
    y = tf.constant(train_y, dtype = tf.float32, shape = train_y.shape)

    # TODO:[YC1.9] Create weights (W) using TF variables
    w = tf.Variable(tf.random.normal([train_x.shape[1], 1], 0, tf.sqrt(2./ 6.)))

    # TODO:[YC1.9] Create a feed-forward operator
    pred = 1./(1. + tf.exp(-tf.matmul(x, w)))

    # TODO:[YC1.9] Write the cost function
    cost = tf.reduce_mean(-(y*tf.math.log(pred) + (1-y)*tf.math.log(1 - pred)), axis = 0)

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005    

    # TODO:[YC1.9] Implement GD
    optimizer = tf.reduce_mean(x*(pred-y), axis = 0)

    # Start training
    for epoch in range(num_epoch):
        with tf.GradientTape() as tape:
            pred = 1. / (1. + tf.exp(-tf.matmul(x, w)))
            cost = tf.reduce_mean(-(y * tf.math.log(pred) + (1 - y) * tf.math.log(1 - pred)), axis = 0)

        gradients = tape.gradient(cost, [w])
        w.assign_sub(learning_rate * gradients[0])

        print(f"Epoch {epoch + 1}: loss is {cost.numpy()[0]:.5f}")

    # TODO:[YC1.9] Compute test result (precision, recall, f1-score)
    precision = 0
    recall = 0
    f1 = 0

    test_x = tf.convert_to_tensor(test_x, dtype=tf.float32)
    test_y = tf.convert_to_tensor(test_y, dtype=tf.float32)

    pred_y = 1. / (1. + tf.exp(-tf.matmul(test_x, w)))
    pred_y = tf.round(pred_y)

    true_positives = tf.reduce_sum(tf.cast((pred_y == 1) & (test_y == 1), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(pred_y == 1, tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(test_y == 1, tf.float32))

    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    f1 = 2 * precision * recall / (precision + recall)

    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1-score: %.3f" % f1)

