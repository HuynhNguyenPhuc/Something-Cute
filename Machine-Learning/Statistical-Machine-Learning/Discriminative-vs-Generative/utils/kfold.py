import numpy as np

class KFold:
    def __init__(self, num_folds, shuffle=False, random_state=42):
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y):
        X_temp = X.copy()
        y_temp = y.copy()

        if self.shuffle:
            np.random.seed(self.random_state)
            indices = np.random.permutation(X_temp.shape[0])
            X_temp = X_temp[indices]
            y_temp = y_temp[indices]

        X_fold = np.array_split(X_temp, self.num_folds)
        y_fold = np.array_split(y_temp, self.num_folds)

        train_Xs = []
        train_ys = []
        val_Xs = []
        val_ys = []

        for i in range(self.num_folds):
            train_X = np.concatenate(X_fold[:i] + X_fold[i+1:], axis = 0)
            val_X = X_fold[i]
            train_y = np.concatenate(y_fold[:i] + y_fold[i+1:], axis = 0)
            val_y = y_fold[i]

            train_Xs.append(train_X)
            train_ys.append(train_y)
            val_Xs.append(val_X)
            val_ys.append(val_y)

        return train_Xs, train_ys, val_Xs, val_ys