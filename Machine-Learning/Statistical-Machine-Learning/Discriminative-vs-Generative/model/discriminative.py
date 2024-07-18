import numpy as np
from keras.utils import to_categorical
from scipy.stats import mode
from utils.kfold import KFold
from model.classifer import Classifier

class DiscriminativeModel(Classifier):
    def __init__(self, data_dir, label = "species"):
        super().__init__(data_dir, label)

        self.W = []
        self.confusion_matrix = []

        dummy_vector = np.ones(self.num_observations).reshape(-1, 1)
        self.X = np.concatenate([self.X, dummy_vector], axis = 1)

        self.num_folds = 5
        self.kfold = KFold(X = self.X, y = self.y, num_folds=self.num_folds, num_observations=self.num_observations)

    def predict(self, W, X):
        y = X @ W
        return y

class OneVsOne(DiscriminativeModel):
    def __init__(self, data_dir, label="species"):
        super().__init__(data_dir, label)

    def convert(self, y, u, v):
        u_vector = (y == u).astype(int).reshape(-1, 1)
        return np.concatenate([1 - u_vector, u_vector], axis = 1)

    def train(self):
        X_train, y_train, X_val, y_val = None, None, None, None
        for i in range(self.num_folds):
            y_pred = []
            X_train, y_train, X_val, y_val = self.kfold.get_train_and_validation_data(i)
            for u in range(self.num_categories):
                for v in range(self.num_categories):
                    if u >= v:
                        continue
                    indices = ((y_train - u)*(y_train - v) == 0)
                    X_train_temp = X_train[indices]
                    y_train_temp = y_train[indices] 
                    y_train_temp = self.convert(y_train_temp, u, v)

                    W = np.linalg.inv(X_train_temp.T @ X_train_temp) @ X_train_temp.T @ y_train_temp
                    self.W.append(W)
                    prediction = np.argmax(self.predict(W, X_val), axis = 1).reshape(-1, 1)
                    prediction = (prediction * u + (1 - prediction) * v).astype(int)

                    y_pred.append(prediction)
                
            y_pred = np.concatenate(y_pred, axis = 1)
            most_frequence_value = mode(y_pred, axis=1).mode

            confusion_matrix = self.get_confusion_matrix(y_val, most_frequence_value)
            self.confusion_matrix.append(confusion_matrix)

class OneVsTheRest(DiscriminativeModel):
    def __init__(self, data_dir, label="species"):
        super().__init__(data_dir, label)

    def train(self):
        X_train, y_train, X_val, y_val = None, None, None, None
        for i in range(self.num_folds):
            y_pred = []
            X_train, y_train, X_val, y_val = self.kfold.get_train_and_validation_data(i)
            for j in range(self.num_categories - 1):
                y_train_temp = np.concatenate([(y_train != j).astype(int).reshape(-1, 1), (y_train == j).astype(int).reshape(-1, 1)], axis = 1)
                W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train_temp
                self.W.append(W)
                prediction = np.argmax(self.predict(W, X_val), axis = 1).reshape(-1, 1)
                y_pred.append(prediction)

            y_pred = np.concatenate(y_pred, axis = 1)
            last_class_pred = (np.sum(y_pred, axis = 1) == 0).astype(int).reshape(-1, 1)
            y_pred = np.concatenate([y_pred, last_class_pred], axis = 1)
            y_pred = np.argmax(y_pred, axis = 1).reshape(-1, 1)
            
            confusion_matrix = self.get_confusion_matrix(y_val, y_pred)
            self.confusion_matrix.append(confusion_matrix)
        
class MultipleClass(DiscriminativeModel):
    def __init__(self, data_dir, label="species"):
        super().__init__(data_dir, label)

    def train(self):
        X_train, y_train, X_val, y_val = None, None, None, None
        for i in range(self.num_folds):
            X_train, y_train, X_val, y_val = self.kfold.get_train_and_validation_data(i)

            y_train = to_categorical(y_train)

            W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
            self.W.append(W)

            y_pred = self.predict(W, X_val)
            y_pred = np.argmax(y_pred, axis = 1).reshape(-1, 1)

            confusion_matrix = self.get_confusion_matrix(y_val, y_pred)
            self.confusion_matrix.append(confusion_matrix)

class Fisher(DiscriminativeModel):
    def __init__(self, data_dir, label="species"):
        super().__init__(data_dir, label)

        self.X = self.X[:,:-1]
        self.X = self.dimensionality_reduction(self.X, self.y)
        dummy_vector = np.ones((self.X.shape[0], 1))
        self.X = np.concatenate([self.X, dummy_vector], axis = 1)

        self.num_folds = 5
        self.kfold = KFold(X = self.X, y = self.y, num_folds=self.num_folds, num_observations=self.num_observations)

    def get_within_class_covariance_matrix(self, partitions):
        return np.sum([partition.shape[0] * self.get_covariance_matrix(partition) for partition in partitions], axis = 0)

    def get_between_class_covariance_matrix(self, mean_vectors, overall_mean, partitions):
        S_B = np.zeros((overall_mean.shape[0], overall_mean.shape[0]))
        for i, mean_vec in enumerate(mean_vectors):
            n = partitions[i].shape[0]
            mean_vec = mean_vec.reshape(-1, 1)
            overall_mean = overall_mean.reshape(-1, 1)
            S_B += n * (mean_vec - overall_mean) @ (mean_vec - overall_mean).T
        return S_B

    
    def dimensionality_reduction(self, X, y, num_dimensions = 2):
        mean_vectors = []
        partitions = self.group_by_class(X, y)

        mean_vectors = [np.mean(partition, axis = 0) for partition in partitions]

        S_W = self.get_within_class_covariance_matrix(partitions)

        overall_mean = np.mean(X, axis=0).reshape(X.shape[1], 1)
        S_B = self.get_between_class_covariance_matrix(mean_vectors, overall_mean, partitions) 
        
        # print("Within-class covariance matrix:\n", S_W)
        # print("Between-class covariance matrix:\n", S_B)

        eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    
        W = np.concatenate([eig_pairs[i][1].reshape(-1, 1) for i in range(num_dimensions)], axis = 1)

        X_lda = X @ W
        return X_lda

    def train(self):
        X_train, y_train, X_val, y_val = None, None, None, None
        for i in range(self.num_folds):
            X_train, y_train, X_val, y_val = self.kfold.get_train_and_validation_data(i)

            # X_train = self.dimensionality_reduction(X_train, y_train)
            # X_val = self.dimensionality_reduction(X_val, y_val)

            # X_train = np.concatenate([X_train, np.ones((X_train.shape[0], 1))], axis = 1)
            # X_val = np.concatenate([X_val, np.ones((X_val.shape[0], 1))], axis = 1)

            y_train = to_categorical(y_train)

            W = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train
            self.W.append(W)

            y_pred = self.predict(W, X_val)
            y_pred = np.argmax(y_pred, axis = 1).reshape(-1, 1)

            confusion_matrix = self.get_confusion_matrix(y_val, y_pred)
            self.confusion_matrix.append(confusion_matrix)
