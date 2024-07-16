import numpy as np
import matplotlib.pyplot as plt

class KMeanClustering:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.centroids = []
        self.labels = []

    def initialize_centroids(self):
        self.centroids = self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]

    def euclidean_distance(self, x1, x2):
        if x1.ndim != x2.ndim:
            raise ValueError('x1 and x2 must have the same dimension')
        return np.sqrt(np.sum((x1 - x2)**2))
    
    def find_closest_centroid(self, x):
        distances = [self.euclidean_distance(x, c) for c in self.centroids]
        return np.argmin(distances)
    
    def update_centroids(self):
        for i in range(self.k):
            self.centroids[i] = np.mean(self.data[self.labels == i], axis=0)
        print("Centroids: ", end="")
        for i in range(self.k):
            print(self.centroids[i], end=" ")
        print()

    def train(self, epochs=100):
        self.initialize_centroids()

        prev_labels = np.zeros(self.data.shape[0])

        for _ in range(epochs):
            # E step
            self.labels = np.array([self.find_closest_centroid(x) for x in self.data])
            # M step
            self.update_centroids()
            if np.all(np.less(np.abs(self.labels - prev_labels), 1e-6)):
                print("Converged")
                break
            prev_labels = self.labels

    def predict(self, x):
        return np.apply_along_axis(func1d=self.find_closest_centroid, axis=1, arr=x).reshape(-1, 1)
    
    def plot(self):
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', marker='x')
        plt.show()
    
if __name__ == '__main__':
    data = np.random.rand(100, 2)
    k = 3
    kmeans = KMeanClustering(data, k)
    kmeans.train()
    kmeans.plot()