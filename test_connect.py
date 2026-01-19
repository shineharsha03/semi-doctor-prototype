import httpx
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

print(f"🔑 Key loaded: {api_key[:5]}... (checking connection)")

try:
    # We try to connect with NO security verification
    response = httpx.get(
        "https://api.openai.com/v1/models", 
        headers={"Authorization": f"Bearer {api_key}"},
        verify=False, 
        timeout=10.0
    )
    print(f"✅ Status Code: {response.status_code}")
    if response.status_code == 200:
        print("🚀 SUCCESS! We can reach OpenAI.")
    else:
        print(f"⚠️ Connected, but got error: {response.text}")
except Exception as e:
    print(f"❌ CRITICAL FAILURE: {e}")