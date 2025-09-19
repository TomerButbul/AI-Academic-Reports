import numpy as np

class Perceptron:
    def __init__(self, num_features, num_classes):
        self.weights = np.zeros((num_classes, num_features))
        self.biases = np.zeros(num_classes)
        self.num_classes = num_classes

    def train(self, X, y, epochs=50, learning_rate=0.1, regularization_strength=0.1):
        for epoch in range(epochs):
            learning_rate_decay = learning_rate / (1 + epoch * 0.01)  
            for i in range(len(X)):
                true_label = y[i]
                scores = np.dot(self.weights, X[i]) + self.biases
                predicted_label = np.argmax(scores)

                if predicted_label != true_label:
                    self.weights[true_label] += learning_rate_decay * X[i]
                    self.biases[true_label] += learning_rate_decay
                    self.weights[predicted_label] -= learning_rate_decay * X[i]
                    self.biases[predicted_label] -= learning_rate_decay

            self.weights -= regularization_strength * self.weights

    def predict(self, X):
        scores = np.dot(X, self.weights.T) + self.biases
        return np.argmax(scores, axis=1)

    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
