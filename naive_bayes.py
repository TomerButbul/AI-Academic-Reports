import numpy as np
from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, num_classes, alpha=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_priors = None
        self.feature_likelihoods = None

    def train(self, X, y):
        num_samples, num_features = X.shape
        self.class_priors = np.zeros(self.num_classes)
        self.feature_likelihoods = defaultdict(lambda: np.zeros(num_features))

        for c in range(self.num_classes):
            X_c = X[y == c]
            self.class_priors[c] = (X_c.shape[0] + self.alpha) / (num_samples + self.num_classes * self.alpha)
            self.feature_likelihoods[c] = (np.sum(X_c, axis=0) + self.alpha) / (X_c.shape[0] + 2 * self.alpha)
            self.feature_likelihoods[c] = np.clip(self.feature_likelihoods[c], 1e-9, 1 - 1e-9)

    def predict(self, X):
        num_samples = X.shape[0]
        predictions = []

        for i in range(num_samples):
            log_posteriors = []

            for c in range(self.num_classes):
                likelihood = self.feature_likelihoods[c]
                log_likelihood = X[i] * np.log(likelihood) + (1 - X[i]) * np.log(1 - likelihood)
                log_posterior = np.log(self.class_priors[c]) + np.sum(log_likelihood)
                log_posteriors.append(log_posterior)

            predictions.append(np.argmax(log_posteriors))

        return np.array(predictions)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
