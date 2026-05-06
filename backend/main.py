from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import numpy as np
from models import gradient_descent_linear, gradient_descent_logistic, sigmoid
from datasets import get_datasets

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATASETS = get_datasets()

def find_dataset(name):
    for category in DATASETS.values():
        if name in category:
            return category[name]
    return None


class TrainRequest(BaseModel):
    dataset_name: str
    learning_rate: float
    n_iterations: int
    test_size: float


@app.get("/datasets")
def get_dataset_info():
    result = {}
    for category, datasets in DATASETS.items():
        result[category] = {}
        for key, val in datasets.items():
            result[category][key] = {
                "name":        val["name"],
                "description": val["description"],
                "model_type":  val["model_type"],
                "X":           val["X"],
                "y":           val["y"]
            }
    return result


@app.post("/train")
def train_model(req: TrainRequest):
    dataset = find_dataset(req.dataset_name)
    if not dataset:
        raise HTTPException(status_code=400, detail="Dataset not found")

    model_type = dataset["model_type"]
    X = np.array(dataset["X"])
    y = np.array(dataset["y"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=req.test_size, random_state=42
    )

    # ── LINEAR ───────────────────────────────────────────────────
    if model_type == "linear":
        theta, cost_history, diverged = gradient_descent_linear(
            X_train, y_train,
            learning_rate=req.learning_rate,
            n_iterations=req.n_iterations
        )

        x_line   = np.linspace(X.min(), X.max(), 100)
        x_line_b = np.c_[np.ones(100), x_line]
        y_line   = x_line_b.dot(np.array(theta))

        return {
            "model_type":   "linear",
            "theta":        theta,
            "cost_history": cost_history,
            "diverged":     diverged,
            "train":        {"X": X_train.tolist(), "y": y_train.tolist()},
            "test":         {"X": X_test.tolist(),  "y": y_test.tolist()},
            "fit_line":     {"x": x_line.tolist(),  "y": y_line.tolist()}
        }

    # ── LOGISTIC ─────────────────────────────────────────────────
    elif model_type == "logistic":
        theta, cost_history, diverged = gradient_descent_logistic(
            X_train, y_train,
            learning_rate=req.learning_rate,
            n_iterations=req.n_iterations
        )

        # Decision boundary mesh grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 80),
            np.linspace(y_min, y_max, 80)
        )
        grid   = np.c_[xx.ravel(), yy.ravel()]
        grid_b = np.c_[np.ones(len(grid)), grid]
        zz     = sigmoid(grid_b.dot(np.array(theta))).reshape(xx.shape)

        # Metrics on test set
        X_test_b    = np.c_[np.ones(len(X_test)), X_test]
        y_pred_prob = sigmoid(X_test_b.dot(np.array(theta)))
        y_pred      = (y_pred_prob >= 0.5).astype(int)
        accuracy    = float(np.mean(y_pred == y_test))

        TN = int(np.sum((y_pred == 0) & (y_test == 0)))
        FP = int(np.sum((y_pred == 1) & (y_test == 0)))
        FN = int(np.sum((y_pred == 0) & (y_test == 1)))
        TP = int(np.sum((y_pred == 1) & (y_test == 1)))

        return {
            "model_type":       "logistic",
            "theta":            theta,
            "cost_history":     cost_history,
            "diverged":         diverged,
            "train":            {"X": X_train.tolist(), "y": y_train.tolist()},
            "test":             {"X": X_test.tolist(),  "y": y_test.tolist()},
            "boundary_grid":    {"xx": xx.tolist(), "yy": yy.tolist(), "zz": zz.tolist()},
            "accuracy":         accuracy,
            "confusion_matrix": [[TN, FP], [FN, TP]]
        }
