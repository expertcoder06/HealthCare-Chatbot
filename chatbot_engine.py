
from symptom_nlp import extract_symptoms
from healthcare_ml import predict_disease


def process_patient_message(age, gender, message):

    # Extract symptoms using NLP
    symptoms = extract_symptoms(message)

    if not symptoms:
        return {
            "status": "no_symptoms",
            "message": "I could not identify any supported symptoms."
        }

    # Convert extracted symptoms into ML input
    symptom_text = ", ".join(symptoms)

    # Predict using the trained ML model
    predictions = predict_disease(
        age=age,
        gender=gender,
        symptoms=symptom_text
    )

    return {
        "status": "success",
        "extracted_symptoms": symptoms,
        "predictions": predictions
    }
