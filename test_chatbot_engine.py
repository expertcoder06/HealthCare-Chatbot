from chatbot_engine import process_patient_message

result = process_patient_message(
    age=35,
    gender="Female",
    message="I have a severe headache, feel dizzy, and sometimes experience nausea."
)

print("Extracted symptoms:", result["extracted_symptoms"])
print("\nPredictions:")

for prediction in result["predictions"]:
    print(
        prediction["Disease"],
        prediction["Model Score (%)"]
    )