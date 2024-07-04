"""bin_classify_np_unittest.py
This file is to test your codes in bin_classify_np.py
Author: Kien Huynh
"""

from bin_classify_np import *
from data_util import get_eclipse_data

def testcase_check(your_arr, test_arr, testname, print_all, print_ind=None):
    eps = 0.00001
    if (type(your_arr) != type(test_arr)):
        print("Testing %s: Failed. Your arr should be %s but it is %s instead." % (testname, type(test_arr), type(your_arr)))
        return True
    
    if (your_arr.shape != test_arr.shape):
        print("Testing %s: Failed. Your arr should have a shape of %s but its shape is %s instead." % (testname, test_arr.shape, your_arr.shape))
        return True

    if (np.sum((your_arr-test_arr)**2) < eps):
        print("Testing %s: Passed." % testname)
    else:
        print("Testing %s: Failed." % testname)
        if (print_all):
            print("Your array is")
            print(your_arr)
            print("While it should be")
            print(test_arr)
        else:
            print("The first few rows of your array are")
            print(your_arr[print_ind, 0])
            print("While they should be")
            print(test_arr[print_ind, 0])
    print("----------------------------------------")
    return False

def unit_test(train_x, train_y):
    """unit_test
    
    Test most functions in this file
    :param train_x: input of train data. Please feed train_x from main into this function.
    :param train_y: label of train data. Please feed train_y from main into this function.
    """
    testcase = np.load('./data/unittest.npy', allow_pickle=True, encoding='latin1')
    testcase = testcase[()]    

    train_x = train_x[0:2, :]
    train_y = train_y[0:2:, :]
     
    train_x1 = add_one(train_x)
    train_x_poly = add_poly_feature(train_x, 2)
 
    if (testcase_check(train_x1, testcase['train_x1'], "add_one", True)):
        return

    if(testcase_check(train_x_poly, testcase['train_x_poly'], "add_poly_feature", True)):
        return

    train_x = testcase['train_x1']

    learning_rate = 0.0001
    momentum_rate = 0.0001

    for i in range(10): 
        test_dict = testcase['output'][i]
        classifier = LogisticClassifier()
        classifier.w = test_dict['w']
        
        y_hat = classifier.feed_forward(train_x)
        if(testcase_check(y_hat, test_dict['y_hat'], "feed_forward %d" % (i+1), True)):
            return

        loss = classifier.compute_loss(train_y, y_hat)
        if(testcase_check(loss, test_dict['loss'], "compute_loss %d" % (i+1), True)):
            return

        grad = classifier.get_grad(train_x, train_y, y_hat)
        if(testcase_check(grad, test_dict['grad'], "get_grad %d" % (i+1), True)):
            return

        classifier.update_weight(grad, 0.001)
        if(testcase_check(classifier.w, test_dict['w_1'], "update_weight %d" % (i+1), True)):
            return
        
        momentum = np.ones_like(test_dict['grad'])
        classifier.update_weight_momentum(grad, learning_rate, momentum, momentum_rate)
        if(testcase_check(classifier.w, test_dict['w_2'], "update_weight_momentum %d" % (i+1), True)):
            return 

if __name__ == "__main__":
    # Load data from file
    # Make sure that eclipse-data.npz is in data/
    train_x, train_y, test_x, test_y = get_eclipse_data()

    unit_test(train_x, train_y)
