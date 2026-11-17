
from symptom_nlp import extract_symptoms

test_cases = [
    "I have a fever and headache.",
    "I do not have fever.",
    "I have headache but no nausea.",
    "I am experiencing dizziness without vomiting."
]

for message in test_cases:

    print("\nPatient:", message)

    print(
        "Extracted:",
        extract_symptoms(message)
    )
