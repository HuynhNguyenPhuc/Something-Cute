import numpy as np
from util import *

class GaussianDistribution:
    @staticmethod
    def pdf(x, mean, covariance, epsilon=1e-9):
        dim = mean.shape[0]
        determinant = np.linalg.det(covariance + epsilon * np.eye(dim))
        inverse_covariance = np.linalg.inv(covariance + epsilon * np.eye(dim))
        diff = x - mean
        exponent = -0.5 * (diff.T @ inverse_covariance @ diff)
        return np.exp(exponent) / (np.sqrt((2 * np.pi) ** dim * determinant))

class GaussianMixtureModel:
    def __init__(self, data, k):
        self.data = data
        self.k = k
        self.pi = []
        self.mu = []
        self.sigma = []

    def initialize_parameters(self):
        self.pi = np.random.rand(self.k)
        self.pi = self.pi / np.sum(self.pi)
        self.mu = self.data[np.random.choice(self.data.shape[0], self.k, replace=False)]
        self.sigma = np.array([np.eye(self.data.shape[1]) for _ in range(self.k)])

    def e_step(self):
        """ Expectation step """
        posterior = np.zeros((self.data.shape[0], self.k))
        for i in range(self.data.shape[0]):
            for j in range(self.k):
                posterior[i][j] = self.pi[j] * GaussianDistribution.pdf(self.data[i].T, self.mu[j].T, self.sigma[j])
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        return posterior
    
    def m_step(self, posterior):
        """ Maximization step """
        self.pi = np.mean(posterior, axis=0)
        self.pi = self.pi / np.sum(self.pi)
        self.mu = np.dot(posterior.T, self.data) / np.sum(posterior, axis=0).reshape(-1, 1)
        self.sigma = np.zeros((self.k, self.data.shape[1], self.data.shape[1]))
        for i in range(self.k):
            self.sigma[i] = np.dot((self.data - self.mu[i]).T, (self.data - self.mu[i]) * posterior[:, i].reshape(-1, 1)) / np.sum(posterior[:, i])

    def log_likelihood(self):
        likelihood = 0
        for i in range(self.data.shape[0]):
            likelihood += np.log(np.sum([self.pi[j] * GaussianDistribution.pdf(self.data[i].T, self.mu[j].T, self.sigma[j]) for j in range(self.k)]))
        return likelihood
    
    def train(self, max_iterations = 100, tol = 1e-6):
        prev_likelihood = -np.inf
        for _ in range(max_iterations):
            posterior = self.e_step()
            self.m_step(posterior)
            likelihood = self.log_likelihood()
            if np.less(np.abs(likelihood - prev_likelihood), tol):
                break
            prev_likelihood = likelihood
    
    def predict_proba(self, x):
        posterior = np.zeros((x.shape[0], self.k))
        for i in range(x.shape[0]):
            for j in range(self.k):
                posterior[i][j] = self.pi[j] * GaussianDistribution.pdf(x[i].T, self.mu[j].T, self.sigma[j])
        posterior = posterior / np.sum(posterior, axis=1, keepdims=True)
        return posterior
    
    def predict(self, x):
        probs = np.zeros((x.shape[0], 1))
        for i in range(x.shape[0]):
            for j in range(self.k):
                probs[i][0] += self.pi[j] * GaussianDistribution.pdf(x[i].T, self.mu[j].T, self.sigma[j])
        return probs
    
if __name__ == '__main__':
    train_x, train_y, test_x, test_y, classes, num_classes = preprocessing()

    gmm_models = []

    for i in range(num_classes):
        gmm = GaussianMixtureModel(train_x[train_y.flatten() == i + 1], k=3)
        gmm.initialize_parameters()
        gmm.train(max_iterations=100)
        gmm_models.append(gmm)

    predictions = np.argmax(np.concatenate([gmm.predict(test_x) for gmm in gmm_models], axis = 1), axis = 1)

    confusion_matrix = np.zeros((num_classes, num_classes))

    for i in range(test_y.shape[0]):
        confusion_matrix[test_y[i] - 1, predictions[i]] += 1

    print("Confusion Matrix:")
    print(confusion_matrix)
    print("Accuracy: ", np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix))