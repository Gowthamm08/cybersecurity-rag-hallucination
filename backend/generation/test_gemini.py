
import os
from dotenv import load_dotenv
from google import genai

# Load API key from the project's .env file
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Check your .env file."
    )

try:
    client = genai.Client(api_key=api_key)

    print("\nGemini API connection successful!")
    print("\nAvailable Gemini models:\n")

    for model in client.models.list():
        print(model.name)

    client.close()

except Exception as error:
    print("\nGemini connection failed.")
    print("Error:", error)
