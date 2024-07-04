import numpy as np

class KFold:
    def __init__(self, X, y, num_folds, num_observations, mode = "random", random_state = 42):
        self.X = X
        self.y = y
        
        self.num_folds = num_folds
        self.num_observations = num_observations

        self.random_state = random_state
        self.indices = self.generate_indices(mode)
        
        self.X_fold = []
        self.y_fold = []
        for i in range(self.num_folds):
            self.X_fold.append(self.X[self.indices[i]])
            self.y_fold.append(self.y[self.indices[i]])

    def generate_indices(self, mode="random"):
        folds = []

        fold_size = self.num_observations // self.num_folds

        if mode == "normal":
            indices = list(range(self.num_observations))
        elif mode == "random":
            np.random.seed(self.random_state)
            indices = np.random.permutation(self.num_observations)

        for i in range(self.num_folds):
            start_index = i * fold_size
            end_index = self.num_observations if i == self.num_folds - 1 else start_index + fold_size
            folds.append(indices[start_index:end_index])

        return folds
    
    def get_train_and_validation_data(self, fold_index):
        X_val = self.X_fold[fold_index]
        y_val = self.y_fold[fold_index]
        
        X_train = np.concatenate(self.X_fold[:fold_index] + self.X_fold[fold_index + 1:])
        y_train = np.concatenate(self.y_fold[:fold_index] + self.y_fold[fold_index + 1:])
        
        return X_train, y_train, X_val, y_val