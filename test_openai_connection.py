
from dotenv import load_dotenv
from openai import OpenAI
import os

# Load API key
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Send a test request
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Explain what a migraine is in two simple sentences."
)

print("\nOpenAI Response:\n")
print(response.output_text)
