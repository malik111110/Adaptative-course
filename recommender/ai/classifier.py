from typing import List
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ..core.models import StudentProfile

class DropoutClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        self.is_trained = False

    def train(self, X: List[List[float]], y: List[bool]) -> None:
        """Train the classifier on preprocessed student data."""
        X_np = np.array(X)
        y_np = np.array([1 if label else 0 for label in y])
        self.model.fit(X_np, y_np)
        self.is_trained = True

    def predict_proba(self, X: List[List[float]]) -> np.ndarray:  # Change return type to np.ndarray
        """Predict dropout probability for a list of students."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before predicting.")
        X_np = np.array(X)
        probabilities = self.model.predict_proba(X_np)[:, 1]  # Probability of dropout (class 1)
        return probabilities  # Return NumPy array, not list

    def predict(self, X: List[List[float]]) -> List[bool]:
        """Predict binary dropout likelihood."""
        if not self.is_trained:
            raise ValueError("Classifier must be trained before predicting.")
        X_np = np.array(X)
        predictions = self.model.predict(X_np)
        return [bool(pred) for pred in predictions]