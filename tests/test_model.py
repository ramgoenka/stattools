import numpy as np
from stattools.predictive_modeling import LogisticRegression  # replace 'your_module' with the actual module name

def test_logistic_regression():
    np.random.seed(42)
    num_samples = 200
    num_features = 4
    X = np.random.randn(num_samples, num_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    logreg = LogisticRegression(learning_rate=0.01, epochs=1000)
    X_train, X_test, y_train, y_test = logreg.split_data(X, y, test_size=0.2)
    logreg.fit(X_train, y_train)
    predictions = logreg.predict(X_test)
    accuracy = (predictions == y_test).mean()
    assert accuracy >= 0.7, f"Accuracy too low."
    assert len(X_train) == 160, "Wrong number of samples for train."
    assert len(X_test) == 40, "Wrong number of samples for test."
    assert len(y_train) == 160, "Wrong number of labels for train"
    assert len(y_test) == 40, "Wrong number of samples for test"
