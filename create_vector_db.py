
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# 1. Extract PDF text
reader = PdfReader("documents/migraine.pdf")

full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# 2. Split into chunks
chunk_size = 500
overlap = 100

chunks = []
start = 0

while start < len(full_text):
    chunk = full_text[start:start + chunk_size].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

# 3. Load embedding model
print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Generate embeddings
embeddings = model.encode(chunks).tolist()

# 5. Create persistent ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name="healthcare_documents"
)

# 6. Store chunks and metadata
collection.add(
    ids=[f"migraine_{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {
            "source": "migraine.pdf",
            "page_type": "healthcare_reference"
        }
        for _ in chunks
    ]
)

print("\nVector database created successfully!")
print("Total chunks stored:", collection.count())
