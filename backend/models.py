import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient_descent_linear(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples = len(y)
    theta = np.zeros(2)
    X_b = np.c_[np.ones(n_samples), X]
    cost_history = []
    diverged = False

    for i in range(n_iterations):
        predictions = X_b.dot(theta)
        errors      = predictions - y
        gradients   = (1 / n_samples) * X_b.T.dot(errors)
        theta       = theta - learning_rate * gradients
        cost        = (1 / (2 * n_samples)) * np.sum(errors ** 2)

        if np.isinf(cost) or np.isnan(cost):
            diverged = True
            break

        cost_history.append(cost)

    return theta.tolist(), [float(c) for c in cost_history], diverged


def gradient_descent_logistic(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples  = len(y)
    n_features = X.shape[1] if X.ndim > 1 else 1
    theta      = np.zeros(n_features + 1)   # +1 for bias
    X_b        = np.c_[np.ones(n_samples), X]
    cost_history = []
    diverged = False

    for i in range(n_iterations):
        z           = X_b.dot(theta)
        predictions = sigmoid(z)
        errors      = predictions - y
        gradients   = (1 / n_samples) * X_b.T.dot(errors)
        theta       = theta - learning_rate * gradients

        # Clip to avoid log(0) blowing up
        p    = np.clip(predictions, 1e-10, 1 - 1e-10)
        cost = -1 / n_samples * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

        if np.isinf(cost) or np.isnan(cost):
            diverged = True
            break

        cost_history.append(cost)

    return theta.tolist(), [float(c) for c in cost_history], diverged
