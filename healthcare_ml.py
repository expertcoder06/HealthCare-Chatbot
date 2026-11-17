
import joblib
import pandas as pd
import numpy as np
import sys

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


# Custom transformer used during model training
class SymptomTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        symptoms = [
            [s.strip().lower() for s in row.split(",")]
            for row in X.iloc[:, 0]
        ]

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit(symptoms)

        return self

    def transform(self, X):
        symptoms = [
            [s.strip().lower() for s in row.split(",")]
            for row in X.iloc[:, 0]
        ]

        return self.mlb.transform(symptoms)


# Register the class in the main Python namespace
# Required for compatibility with the model saved in Colab
sys.modules["__main__"].SymptomTransformer = SymptomTransformer


# Load trained model
loaded_model = joblib.load("healthcare_disease_model.joblib")


def predict_disease(age, gender, symptoms):

    patient_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Symptoms": symptoms
    }])

    probabilities = loaded_model.predict_proba(patient_data)[0]

    top_indices = np.argsort(probabilities)[::-1][:3]

    results = []

    for index in top_indices:
        results.append({
            "Disease": loaded_model.classes_[index],
            "Model Score (%)": round(
                probabilities[index] * 100, 2
            )
        })

    return results
