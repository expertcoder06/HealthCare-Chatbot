
from sentence_transformers import SentenceTransformer
import chromadb

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to existing database
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(
    name="healthcare_documents"
)

def retrieve_information(question, top_k=3):

    # Convert question into embedding
    question_embedding = model.encode(question).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
    )

    return results


# Test retrieval
question = "Why does migraine cause nausea and throbbing pain?"
results = retrieve_information(question)

print("\nUser Question:", question)
print("\nRetrieved Information:\n")

for i, document in enumerate(results["documents"][0], start=1):

    print(f"\n--- Result {i} ---")
    print(document)

    print("Source:", results["metadatas"][0][i-1]["source"])
