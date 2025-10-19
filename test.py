import unittest
import os
import joblib
import pandas as pd
import sys
import io
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from train import train

LOG_FILE = "test_log.txt"

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")
    print(msg)

class TestIrisModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.stdout = sys.stderr = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        if not os.path.exists("model/iris_model.pkl"):
            log("Model not found, training...")
            train()
        cls.model = joblib.load("model/iris_model.pkl")
        cls.data = load_iris()

    def test_model_file_exists(self):
        """Check if model file is created"""
        self.assertTrue(os.path.exists("model/iris_model.pkl"), "Model file missing.")

    def test_model_prediction_shape(self):
        """Check if model predictions have correct shape"""
        preds = self.model.predict(self.data.data[:5])
        self.assertEqual(len(preds), 5, "Prediction length mismatch.")

    def test_model_accuracy_above_threshold(self):
        """Check model accuracy threshold"""
        X, y = self.data.data, self.data.target
        preds = self.model.predict(X)
        acc = accuracy_score(y, preds)
        log(f"Accuracy on full dataset: {acc}")
        self.assertGreater(acc, 0.85, "Model accuracy below 0.85")

    def test_model_output_values(self):
        """Check that predicted labels are within expected range"""
        preds = self.model.predict(self.data.data)
        valid_labels = set(self.data.target)
        self.assertTrue(set(preds).issubset(valid_labels), "Predictions contain invalid labels")

    def test_model_inference_no_crash(self):
        """Ensure inference runs without error"""
        try:
            _ = self.model.predict(self.data.data[:1])
            success = True
        except Exception as e:
            success = False
            log(f"Inference error: {e}")
        self.assertTrue(success, "Inference crashed unexpectedly")

if __name__ == '__main__':
    log("Starting tests...")
    unittest.main(verbosity=2)

