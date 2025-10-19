import os
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

LOG_FILE = "train_log.txt"

def log(message: str):
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    print(message)

def train():
    try:
        log("Loading Iris dataset...")
        iris = load_iris()
        X = iris.data
        y = iris.target

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        log(f"Model trained successfully. Accuracy: {acc:.4f}")

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, "model/iris_model.pkl")
        log("Model saved to model/iris_model.pkl")

        return acc
    except Exception as e:
        log(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    train()

