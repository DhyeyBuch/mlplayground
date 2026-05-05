from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.model_selection import train_test_split
import numpy as np
from regression import gradient_descent
from datasets import get_datasets

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup, store in memory
DATASETS = get_datasets()


# --- Pydantic model for /train request body ---

class TrainRequest(BaseModel):
    dataset_name: str
    learning_rate: float
    n_iterations: int
    test_size: float          # e.g. 0.2 means 80/20 split


# --- Endpoint 1: GET /datasets ---

@app.get("/datasets")
def get_dataset_info():
    result = {}
    for key, val in DATASETS.items():
        result[key] = {
            "name": val["name"],
            "description": val["description"],
            "X": val["X"],
            "y": val["y"]
        }
    return result


# --- Endpoint 2: POST /train ---

@app.post("/train")
def train_model(req: TrainRequest):

    if req.dataset_name not in DATASETS:
        raise HTTPException(status_code=400, detail="Dataset not found")

    dataset = DATASETS[req.dataset_name]
    X = np.array(dataset["X"])
    y = np.array(dataset["y"])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=req.test_size,
        random_state=42
    )

    # Run gradient descent on training set only
    theta, cost_history, diverged = gradient_descent(
        X_train,
        y_train,
        learning_rate=req.learning_rate,
        n_iterations=req.n_iterations
    )

    # Generate fitted line for plotting (across full X range)
    x_line = np.linspace(X.min(), X.max(), 100)
    x_line_b = np.c_[np.ones(100), x_line]
    y_line = x_line_b.dot(np.array(theta))

    return {
        "theta": theta,
        "cost_history": cost_history,
        "diverged": diverged,
        "train": {
            "X": X_train.tolist(),
            "y": y_train.tolist()
        },
        "test": {
            "X": X_test.tolist(),
            "y": y_test.tolist()
        },
        "fit_line": {
            "x": x_line.tolist(),
            "y": y_line.tolist()
        }
    }
