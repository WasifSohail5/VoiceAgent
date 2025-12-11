import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os. getenv("OPENAI_API_KEY")
print(f"API Key (first 20 chars): {api_key[:20] if api_key else 'NOT FOUND'}...")
print(f"API Key length: {len(api_key) if api_key else 0}")

# Test basic API
client = OpenAI(api_key=api_key)

try:
    response = client.chat. completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    print(f"✅ API Key is VALID!")
    print(f"Response: {response.choices[0]. message.content}")
except Exception as e:
    print(f"❌ API Key Error: {e}")