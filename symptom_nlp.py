
import re

# Symptoms supported by our trained ML model
SYMPTOM_SYNONYMS = {
    "abdominal pain": ["stomach pain", "belly pain", "abdominal pain"],
    "anxiety": ["anxiety", "feeling anxious"],
    "appetite loss": ["loss of appetite", "poor appetite"],
    "back pain": ["back pain", "backache"],
    "blurred vision": ["blurred vision", "blurry vision"],
    "chest pain": ["chest pain", "pain in chest"],
    "cough": ["cough", "coughing"],
    "depression": ["depression", "feeling depressed"],
    "diarrhea": ["diarrhea", "loose motion", "loose motions"],
    "dizziness": ["dizziness", "dizzy", "lightheaded"],
    "fatigue": ["fatigue", "tiredness", "exhaustion"],
    "fever": ["fever", "high temperature"],
    "headache": ["headache", "head pain"],
    "insomnia": ["insomnia", "difficulty sleeping"],
    "joint pain": ["joint pain", "pain in joints"],
    "muscle pain": ["muscle pain", "body ache", "muscle ache"],
    "nausea": ["nausea", "feeling nauseous", "feeling sick"],
    "rash": ["rash", "skin rash"],
    "runny nose": ["runny nose", "nasal discharge"],
    "shortness of breath": [
        "shortness of breath",
        "difficulty breathing",
        "breathlessness"
    ],
    "sneezing": ["sneezing", "sneeze"],
    "sore throat": ["sore throat", "throat pain"],
    "sweating": ["sweating", "excessive sweating"],
    "swelling": ["swelling", "swollen"],
    "tremors": ["tremors", "shaking", "trembling"],
    "vomiting": ["vomiting", "throwing up", "threw up"],
    "weight gain": ["weight gain", "gaining weight"],
    "weight loss": ["weight loss", "losing weight"]
}

NEGATION_WORDS = {
    "no", "not", "without", "never",
    "don't", "doesn't", "didn't",
    "haven't", "hasn't", "hadn't",
    "denies", "denied"
}


def extract_symptoms(text):

    text = text.lower()

    # Split text into separate clauses
    clauses = re.split(
        r"\b(?:but|however|although|whereas)\b|[,.;!?]",
        text
    )

    detected_symptoms = []

    for clause in clauses:

        words = clause.split()

        for symptom, variations in SYMPTOM_SYNONYMS.items():

            for variation in variations:

                pattern = r"\b" + re.escape(variation) + r"\b"

                match = re.search(pattern, clause)

                if match:

                    preceding_text = clause[:match.start()]
                    preceding_words = preceding_text.split()

                    # Check the last three words before the symptom
                    recent_words = preceding_words[-3:]

                    is_negated = any(
                        word in NEGATION_WORDS
                        for word in recent_words
                    )

                    if not is_negated and symptom not in detected_symptoms:
                        detected_symptoms.append(symptom)

                    break

    return detected_symptoms
