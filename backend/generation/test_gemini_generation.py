
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY missing from .env")

url = (
    "https://generativelanguage.googleapis.com/"
  "v1beta/models/gemini-3.5-flash-lite:generateContent"
)

headers = {
    "Content-Type": "application/json",
    "x-goog-api-key": api_key
}

payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": "Explain SQL injection in two simple sentences."
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.2,
        "maxOutputTokens": 150
    }
}

try:
    print("Connecting to Gemini...", flush=True)

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=(10, 30)
    )

    print("HTTP status:", response.status_code, flush=True)

    if response.ok:
        result = response.json()
        answer = result["candidates"][0]["content"]["parts"][0]["text"]
        print("\nGemini response:\n")
        print(answer)
    else:
        print("API error:", response.text)

except requests.exceptions.Timeout:
    print("Request timed out after 30 seconds.")

except requests.exceptions.ConnectionError as error:
    print("Network connection error:", error)

except Exception as error:
    print("Unexpected error:", error)
