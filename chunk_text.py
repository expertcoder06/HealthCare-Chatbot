
from pypdf import PdfReader

pdf_path = "documents/migraine.pdf"

reader = PdfReader(pdf_path)

# Extract all text
full_text = ""

for page in reader.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Split text into chunks
chunk_size = 500
overlap = 100

chunks = []

start = 0

while start < len(full_text):
    end = start + chunk_size

    chunk = full_text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - overlap

print("Total characters:", len(full_text))
print("Total chunks:", len(chunks))

print("\nFirst chunk:\n")
print(chunks[0])
