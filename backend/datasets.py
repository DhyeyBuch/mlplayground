import numpy as np
from sklearn.datasets import (load_diabetes, make_regression,
                               make_moons, make_circles, make_classification)

def get_datasets():

    # ── REGRESSION ──────────────────────────────────────────────

    diabetes  = load_diabetes()
    X1        = diabetes.data[:, 2]
    y1        = diabetes.target
    X1        = (X1 - X1.mean()) / X1.std()

    X2, y2    = make_regression(n_samples=100, n_features=1,
                                 noise=15, random_state=42)
    X2        = X2.flatten()
    X2        = (X2 - X2.mean()) / X2.std()

    np.random.seed(42)
    X3        = np.linspace(-3, 3, 100)
    y3        = 2 * X3**2 + X3 + np.random.normal(0, 1, 100)
    X3        = (X3 - X3.mean()) / X3.std()

    # ── CLASSIFICATION ───────────────────────────────────────────

    X4, y4    = make_classification(n_samples=200, n_features=2,
                                     n_redundant=0, n_informative=2,
                                     random_state=42, n_clusters_per_class=1)

    X5, y5    = make_moons(n_samples=200, noise=0.2, random_state=42)

    X6, y6    = make_circles(n_samples=200, noise=0.1,
                              factor=0.5, random_state=42)

    return {
        "regression": {
            "diabetes": {
                "name": "Diabetes Dataset",
                "description": "Classic medical dataset. Strong linear relationship.",
                "model_type": "linear",
                "X": X1.tolist(),
                "y": y1.tolist()
            },
            "synthetic_linear": {
                "name": "Synthetic Linear Data",
                "description": "Clean generated data. Ideal linear fit.",
                "model_type": "linear",
                "X": X2.tolist(),
                "y": y2.tolist()
            },
            "nonlinear": {
                "name": "Quadratic Dataset",
                "description": "Curved pattern. Watch linear regression struggle.",
                "model_type": "linear",
                "X": X3.tolist(),
                "y": y3.tolist()
            }
        },
        "classification": {
            "make_classification": {
                "name": "Linearly Separable",
                "description": "Clean synthetic data. Logistic regression thrives here.",
                "model_type": "logistic",
                "X": X4.tolist(),
                "y": y4.tolist()
            },
            "make_moons": {
                "name": "Make Moons",
                "description": "Non-linear boundary. Watch logistic regression struggle.",
                "model_type": "logistic",
                "X": X5.tolist(),
                "y": y5.tolist()
            },
            "make_circles": {
                "name": "Make Circles",
                "description": "Circular boundary. A hard challenge for a linear classifier.",
                "model_type": "logistic",
                "X": X6.tolist(),
                "y": y6.tolist()
            }
        }
    }
