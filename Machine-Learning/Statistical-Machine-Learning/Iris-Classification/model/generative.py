import numpy as np
from utils.kfold import KFold
from model.classifer import Classifier

class GenerativeModel(Classifier):
    def __init__(self, data_dir, label = "species"):
        super().__init__(data_dir, label)

        self.w = []
        self.bias = []
        self.confusion_matrix = []

        self.num_folds = 5
        self.kfold = KFold(X = self.X, y = self.y, num_folds=self.num_folds, num_observations=self.num_observations)

    def predict(self, X, w, b):
        return X @ w.T + b

    def get_coefficients(self, X, y):
        num_records = X.shape[0]
        partitions = self.group_by_class(X, y)
        mean_vector = np.array([np.mean(partition, axis=0).reshape(-1, 1) for partition in partitions])
        prob_vector = []
        S = []

        for i in range(self.num_categories):
            center = partitions[i] - mean_vector[i].T
            S.append(self.get_covariance_matrix(center))
            prob_vector.append(partitions[i].shape[0] / num_records)

        sigma = np.sum([S_item * prob_item for S_item, prob_item in zip(S, prob_vector)], axis=0)
        sigma_inv = np.linalg.inv(sigma)

        w = np.array([sigma_inv @ m for m in mean_vector]).reshape(self.num_categories, -1)
        bias = np.array([-0.5 * (m.T @ sigma_inv @ m) + np.log(p) for m, p in zip(mean_vector, prob_vector)]).reshape(-1)

        return w, bias


class Bayesian(GenerativeModel):
    def __init__(self, data_dir, label="species"):
        super().__init__(data_dir, label)

    def softmax(self, y):
        exp_y = np.exp(y)
        sum_y = np.sum(exp_y, axis = 1, keepdims = True)
        return exp_y / sum_y 

    def train(self):
        X_train, y_train, X_val, y_val = None, None, None, None
        for i in range(self.num_folds):
            X_train, y_train, X_val, y_val = self.kfold.get_train_and_validation_data(i)
            X_train = X_train[:,:-1]
            X_val = X_val[:,:-1]

            w, bias = self.get_coefficients(X_train, y_train)

            y_pred = self.predict(X_val, w, bias)
            y_pred = self.softmax(y_pred)
            y_pred = np.argmax(y_pred, axis = 1).reshape(-1, 1)

            confusion_matrix = self.get_confusion_matrix(y_val, y_pred)
            self.confusion_matrix.append(confusion_matrix)