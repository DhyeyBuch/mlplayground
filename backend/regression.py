import numpy as np

def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples = len(y)
    theta = np.zeros(2)
    X_b = np.c_[np.ones(n_samples), X]
    
    cost_history = []
    diverged = False

    for i in range(n_iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        
        gradients = (1 / n_samples) * X_b.T.dot(errors)
        theta = theta - learning_rate * gradients
        
        cost = (1 / (2 * n_samples)) * np.sum(errors ** 2)

        if np.isinf(cost) or np.isnan(cost):
            diverged = True
            break

        cost_history.append(cost)

    return theta.tolist(), [float(c) for c in cost_history], diverged
