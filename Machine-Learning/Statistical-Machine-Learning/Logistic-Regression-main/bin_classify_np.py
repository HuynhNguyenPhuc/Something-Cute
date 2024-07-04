"""bin_classify_np.py
This file is for binary classification
Author: Kien Huynh
"""

import numpy as np
import matplotlib.pyplot as plt
from data_util import get_eclipse_data

class MachineLearningAlgorithm(object):
	
    def __init__(self):
        pass

    def train_model(self, x):
        pass

    def test(self, xtest):
        pass
	
class LogisticClassifier(object):
    def __init__(self, w_shape=(3,1)):
        """__init__
        
        :param w_shape: create w with shape w_shape using normal distribution
        """

        # mean = 0
        # std = 1
        self.w = np.random.normal(0, np.sqrt(2./np.sum(w_shape)), w_shape)
        # print(self.w.shape)
        # print(type(self.w))
		
    def feed_forward(self, x):
        """feed_forward
        TODO:[YC1.2]
        This function compute the output of your logistic classification model
        
        :param x: input
        """
        z = np.dot(x, self.w) #x.shape 3000*3 x 3*1
        result = 1./ (1. + np.exp(-z)) 
        return result #print(result.shape) --> (3000*1) - y_training sample

    def compute_loss(self, y, y_hat):
        """compute_loss
        TODO:[YC1.3]
        Compute the loss using y (label) and y_hat (predicted class)

        :param y:  the label, the actual class of the samples
        :param y_hat: the propabilitis that the given samples belong to class 1
        """
        loss = -(y*np.log(y_hat) + (1-y)*np.log(1 - y_hat))
        return np.mean(loss, axis = 0)[0]

    def get_grad(self, x, y, y_hat):
        """get_grad
        TODO:[YC1.4]
        Compute and return the gradient of w

        :param loss: computed loss between y_hat and y in the train dataset
        :param y_hat: predicted y
        """
        w_grad = np.mean(x*(y_hat-y), axis = 0)

        return w_grad.reshape(-1, 1)

    def update_weight(self, grad, learning_rate):
        """update_weight
        TODO:[YC1.5]
        Update w using the computed gradient

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        """
        
        self.w = self.w - learning_rate * grad;
    
    def update_weight_momentum(self, grad, learning_rate, momentum, momentum_rate):
        """update_weight with momentum
        BONUS:[YC1.8]
        Update w using the algorithm with momnetum

        :param grad: gradient computed from the loss
        :param learning_rate: float, learning rate
        :param momentum: the array storing momentum for training w, should have the same shape as w
        :param momentum_rate: float, how much momentum to reuse after each loop (denoted as gamma in the document)
        """
        momentum = momentum_rate * momentum + learning_rate * grad
        self.w = self.w - momentum

    def numerical_check(self, x, y, grad):
        """numerical_check
        This function performs a numerical gradient check
        When you run this function, the grad parameter and the numerical_grad should be close to each other. grad should be around numerical_grad +- 1e-6

        :param x: input
        :param y: label of the input
        :param grad: computed gradient to be check
        """
        eps = 0.000005
        w_test0 = np.copy(self.w)
        w_test1 = np.copy(self.w)
        w_test0[2] = w_test0[2] - eps
        w_test1[2] = w_test1[2] + eps

        y_hat0 = np.dot(x, w_test0)
        y_hat0 = 1. / (1. + np.exp(-y_hat0))
        loss0 = self.compute_loss(y, y_hat0) 

        y_hat1 = np.dot(x, w_test1)
        y_hat1 = 1. / (1. + np.exp(-y_hat1))
        loss1 = self.compute_loss(y, y_hat1) 

        numerical_grad = (loss1 - loss0)/(2*eps)
        print(numerical_grad)
        print(grad[2])
        breakpoint()

def visualize_data(x, y, y_hat):
    """visualize_data
    
    This funciton scatter data points (in x) and color them according to y and y_hat for comparison
    Both figures should be similar
    :param x:
    :param y:
    :param y_hat:
    """
    c = np.copy(y_hat)
    c[c<0.5] = 0
    c[c>0.5] = 1

    rgb = np.eye(3)
    fig = plt.figure(1, figsize = (12,6))
    plt.clf()
    ax.set_title("Actual classes")
    ax.scatter(x[:,0], x[:,1], color = rgb[y[:,0].astype(np.int32),:])
    ax = plt.subplot(1,2,2)
    ax.set_title("Prediction")
    ax.scatter(x[:,0], x[:,1], color = rgb[c[:,0].astype(np.int32),:])
    plt.axis('equal')
    plt.ion() 
    plt.draw()
    plt.show()

def add_poly_feature(x, degree=2):
    """add_feature
    BONUS:[YC1.7]
    This function adds more polynomial feature to x
    :param x: input data
    """
    c0 = x[:, 0].reshape(-1, 1)
    c1 = x[:, 1].reshape(-1, 1)
    return np.concatenate([c0, c1, c0 * c0, c0 * c1, c1 * c1], axis = 1)

def add_one(x):
    """add_one
    TODO:[YC1.1]
    This function add ones as an additional feature for x
    :param x: input data
    """
    dump_vector = np.ones((x.shape[0], 1))
    x = np.concatenate([x, dump_vector], axis = 1)
    return x


def test(classifier, test_x, test_y):
    """test
    TODO:[YC1.6]
    Compute precision, recall and F1-score based on test samples
    :param classifier: the trained classifier
    :param test_x: test samples
    :param test_y: test labels
    """
    precision = 0
    recall = 0
    f1 = 0

    test_y = test_y.astype(int)
    pred_y = classifier.feed_forward(test_x)
    pred_y = (pred_y >= 0.5).astype(int)

    confusion_matrix = np.zeros((2, 2), dtype=int)
    for i in range(test_y.shape[0]):
        confusion_matrix[test_y[i, 0], pred_y[i, 0]] += 1

    precision = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[1, 0])
    recall = confusion_matrix[1, 1]/(confusion_matrix[1, 1] + confusion_matrix[0, 1])

    f1 = 2. * precision * recall / (precision + recall)

    print("Precision: %.3f" % precision)
    print("Recall: %.3f" % recall)
    print("F1-score: %.3f" % f1)

    return (precision, recall, f1)


if __name__ == "__main__":
    # Random seed is fixed so that every run is the same
    # This makes it easier to debug
    np.random.seed(2017)

    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_eclipse_data()
    num_train = train_x.shape[0]
    num_test = test_x.shape[0]  
    
    # Uncomment the following line to run unit_test and check your work
    # unit_test(train_x, train_y)

    # Add more features to train_x and test_x
    train_x = add_poly_feature(train_x, 2)
    test_x = add_poly_feature(test_x, 2)
    
    # Pad 1 as the third feature of train_x and test_x
    train_x = add_one(train_x) 
    test_x = add_one(test_x)
    
    # Create classifier
    num_feature = train_x.shape[1]
    bin_classifier = LogisticClassifier((num_feature, 1)) 

    # Define hyper-parameters and train-related parameters
    num_epoch = 10000
    learning_rate = 0.005
    draw_frequency = 50000

    momentum = 0.

    for e in range(num_epoch):    
        y_hat = bin_classifier.feed_forward(train_x) #sigmoid y_hat (3000*1) 
        loss = bin_classifier.compute_loss(train_y, y_hat)
        grad = bin_classifier.get_grad(train_x, train_y, y_hat)
        # bin_classifier.numerical_check(train_x, train_y, grad)
        bin_classifier.update_weight(grad, learning_rate)
        # bin_classifier.update_weight_momentum(grad, learning_rate, momentum, 0.9)
       
        # Visualizing our data and the classifier predictions every [draw_frequency] epochs
        if (e % draw_frequency == draw_frequency-1):
            visualize_data(train_x, train_y, y_hat) 
        print("Epoch %d: loss is %.5f" % (e+1, loss))

    test(bin_classifier, test_x, test_y)
