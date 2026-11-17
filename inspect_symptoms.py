
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer


# Required to load the model saved from Google Colab
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


# Import after defining the compatibility class
from healthcare_ml import loaded_model


preprocessor = loaded_model.named_steps["preprocessing"]

symptom_transformer = (
    preprocessor.named_transformers_["symptoms"]
)

symptom_names = symptom_transformer.mlb.classes_

print("Total symptoms learned:", len(symptom_names))
print("\nSymptoms available in the ML model:\n")

for index, symptom in enumerate(symptom_names, start=1):
    print(f"{index}. {symptom}")
