import numpy as np

class LogisticRegression:
    """
    A simple logistic regression model for binary classification.

    Methods include:

    - fit: Train the model using gradient descent.

    - predict: Make predictions on new data.
    
    - split_data: Split the dataset into training and testing sets.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def split_data(self, X, y, test_size=0.2, random_state=42):
        np.random.seed(random_state)
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        split_idx = int(X.shape[0] * (1 - test_size))
        
        X_train, X_test = X[indices[:split_idx]], X[indices[split_idx:]]
        y_train, y_test = y[indices[:split_idx]], y[indices[split_idx:]]
        
        return X_train, X_test, y_train, y_test

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.sum(y_pred - y) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.5):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear_model)
        return (y_pred >= threshold).astype(int)