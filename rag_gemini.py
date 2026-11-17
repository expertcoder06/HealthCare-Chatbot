
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb
import os

# 1. Load API key
load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

# 2. Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Connect to ChromaDB
db_client = chromadb.PersistentClient(path="./chroma_db")

collection = db_client.get_collection(
    name="healthcare_documents"
)

# 4. Retrieve relevant information
def retrieve_information(question, top_k=3):

    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results["documents"][0], results["metadatas"][0]


# 5. Generate answer using Gemini
def generate_answer(question):

    documents, metadata = retrieve_information(question)

    context = ""

    for i, document in enumerate(documents):
        context += f"\nSource: {metadata[i]['source']}\n"
        context += document + "\n"

    prompt = f"""
You are a healthcare information assistant.

Answer the user's question using the provided reference context.

Rules:
1. Use the reference context as your primary source.
2. Do not invent medical facts.
3. If the answer is not available in the context, clearly say so.
4. Explain in simple language.
5. Do not diagnose or prescribe medication.
6. Encourage users to consult a qualified healthcare professional.
7. This information is educational, not a medical diagnosis.

Reference Context:
{context}

User Question:
{question}

Answer:
"""


    # Try alternative models if one is temporarily unavailable
    models_to_try = [
        "gemini-flash-lite-latest",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model_name in models_to_try:

        try:
            print(f"Trying Gemini model: {model_name}")

            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )

            if response.text:
                print(f"Successfully used: {model_name}")
                return response.text, metadata

        except Exception as error:
            print(f"{model_name} failed: {error}")
            last_error = error

    raise RuntimeError(
        f"All available Gemini models failed. Last error: {last_error}"
    )



# 6. Test RAG chatbot
if __name__ == "__main__":

    question = "What are the common symptoms of migraine?"

    answer, sources = generate_answer(question)

    print("\nUser Question:")
    print(question)

    print("\nGemini RAG Answer:")
    print(answer)

    print("\nSources:")

    for source in sources:
        print("-", source["source"])
