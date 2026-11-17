
from pypdf import PdfReader

pdf_path = "documents/migraine.pdf"

reader = PdfReader(pdf_path)

print("Total pages:", len(reader.pages))
print("\nExtracted text:\n")

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        print(f"\n--- Page {page_number} ---\n")
        print(text[:1500])
