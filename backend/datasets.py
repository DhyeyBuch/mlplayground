import numpy as np
from sklearn.datasets import load_diabetes, make_regression

def get_datasets():

    # Dataset 1 - Diabetes (classic linear regression dataset)
    diabetes = load_diabetes()
    X1 = diabetes.data[:, 2]  # Single feature for clean 2D plot
    y1 = diabetes.target
    X1 = (X1 - X1.mean()) / X1.std()  # Normalize

    # Dataset 2 - Synthetic linear data (clean, obvious linear pattern)
    X2, y2 = make_regression(
        n_samples=100,
        n_features=1,
        noise=15,
        random_state=42
    )
    X2 = X2.flatten()
    X2 = (X2 - X2.mean()) / X2.std()

    # Dataset 3 - Non-linear (quadratic pattern, linear regression will fail visibly)
    np.random.seed(42)
    X3 = np.linspace(-3, 3, 100)
    y3 = 2 * X3**2 + X3 + np.random.normal(0, 1, 100)
    X3 = (X3 - X3.mean()) / X3.std()

    return {
        "diabetes": {
            "name": "Diabetes Dataset",
            "description": "Classic medical dataset. Strong linear relationship.",
            "X": X1.tolist(),
            "y": y1.tolist()
        },
        "synthetic_linear": {
            "name": "Synthetic Linear Data",
            "description": "Clean generated data. Ideal linear fit.",
            "X": X2.tolist(),
            "y": y2.tolist()
        },
        "nonlinear": {
            "name": "Quadratic Dataset",
            "description": "Curved pattern. Watch linear regression struggle.",
            "X": X3.tolist(),
            "y": y3.tolist()
        }
    }
