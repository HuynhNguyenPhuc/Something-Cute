import numpy as np
import pandas as pd

class Classifier:
    def __init__(self, data_dir, label="species"):
        self.dataset = pd.read_csv(data_dir)
        self.num_observations = self.dataset.shape[0]

        self.dataset[label] = self.dataset[label].astype("category")
        self.dataset["category"] = self.dataset[label].cat.codes
        
        self.X = self.dataset.drop([label, "category"], axis = 1).values
        self.y = self.dataset["category"].values

        self.categories = self.dataset[label].cat.categories
        self.num_categories = len(self.categories)

        print(self.categories)

        self.accuracy = None

    def group_by_class(self, X, y):
        results = []
        for i in range(self.num_categories):
            results.append(X[y == i])
        return results
    
    def get_covariance_matrix(self, X):
        return 1/X.shape[0] * (X.T @ X)
    
    def get_confusion_matrix(self, y_true, y_pred):
        y_true = y_true.reshape(-1, )
        y_pred = y_pred.reshape(-1, )

        confusion_matrix = np.zeros((self.num_categories, self.num_categories), dtype=np.int32)

        for i in range(len(y_true)):
            confusion_matrix[y_true[i], y_pred[i]] += 1

        return confusion_matrix
    
    def print_evaluation(self):
        confusion_matrix = np.sum(self.confusion_matrix, axis=0)
        
        if self.accuracy is None:
            self.accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
        
        print("Accuracy: ", self.accuracy)
        print("Confusion Matrix:\n", confusion_matrix)