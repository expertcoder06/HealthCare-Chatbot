# SQLite fix for Streamlit Cloud (ChromaDB needs SQLite >= 3.35)
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st
from chatbot_engine import process_patient_message

# Page configuration
st.set_page_config(
    page_title="Healthcare AI Chatbot",
    page_icon="🩺",
    layout="centered"
)

# Application heading
st.title("🩺 Healthcare AI Chatbot")
st.write(
    "Your AI-powered healthcare assistant using "
    "NLP, Machine Learning, RAG and Gemini."
)

st.warning(
    "Educational prototype only. This tool does not provide "
    "medical diagnoses or replace professional medical advice."
)

# ==========================================
# SECTION 1: ML + NLP SYMPTOM ANALYSIS
# ==========================================

st.header("1. Symptom Analysis")
st.write("Describe how you are feeling to get possible ML predictions.")

st.subheader("Patient Information")

age = st.number_input(
    "Enter your age",
    min_value=1,
    max_value=120,
    value=25
)

gender = st.selectbox(
    "Select your gender",
    ["Male", "Female", "Other"]
)

st.divider()

st.subheader("Describe Your Symptoms")

message = st.text_area(
    "How are you feeling?",
    placeholder=(
        "Example: I have a severe headache, "
        "feel dizzy, and sometimes experience nausea."
    ),
    height=130
)

if st.button("Analyze My Symptoms", type="primary"):

    if not message.strip():
        st.error("Please describe your symptoms.")

    else:
        try:
            with st.spinner("Understanding your symptoms..."):

                result = process_patient_message(
                    age=age,
                    gender=gender,
                    message=message
                )

            if result["status"] == "no_symptoms":
                st.warning(result["message"])

            else:
                st.success("Symptom analysis completed.")

                st.subheader("Symptoms Identified by NLP")

                for symptom in result["extracted_symptoms"]:
                    st.write(f"✓ {symptom.title()}")

                st.subheader("ML Model Predictions")

                for index, prediction in enumerate(
                    result["predictions"], start=1
                ):
                    st.write(
                        f"**{index}. {prediction['Disease']}**"
                    )

                    st.caption(
                        f"Classifier score: "
                        f"{prediction['Model Score (%)']}%"
                    )

                st.info(
                    "These scores are outputs of a machine-learning "
                    "classifier, not probabilities of having a disease. "
                    "Consult a qualified healthcare professional."
                )

        except Exception as e:
            st.error(f"Symptom analysis failed: {e}")


# ==========================================
# SECTION 2: RAG + GEMINI HEALTHCARE Q&A
# ==========================================

st.divider()

st.header("2. Healthcare Knowledge Assistant")

st.write(
    "Ask general healthcare questions. Gemini will use "
    "information retrieved from your healthcare PDF documents."
)

question = st.text_area(
    "Ask your healthcare question",
    placeholder=(
        "Example: What are the common symptoms of migraine?"
    ),
    height=100,
    key="healthcare_question"
)

if st.button("Ask Healthcare Assistant"):

    if not question.strip():
        st.error("Please enter a question.")

    else:
        try:
            with st.spinner(
                "Searching healthcare documents and generating an answer..."
            ):

                from rag_gemini import generate_answer

                answer, sources = generate_answer(question)

            st.subheader("Gemini's Answer")
            st.write(answer)

            # Display unique source documents
            st.subheader("Reference Documents")

            unique_sources = list(dict.fromkeys(
                source["source"] for source in sources
            ))

            for source in unique_sources:
                st.write(f"📄 {source}")

        except Exception as e:
            st.error(f"Healthcare Q&A failed: {e}")


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "Healthcare AI Chatbot | NLP + ML + RAG + Google Gemini"
)