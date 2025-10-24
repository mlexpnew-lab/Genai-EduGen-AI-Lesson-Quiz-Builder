import os, google.generativeai as genai
from dotenv import load_dotenv
load_dotenv(override=True)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = os.getenv("MODEL_NAME","models/gemini-2.5-flash")

print("DEBUG genai version =", getattr(genai, "__version__", "unknown"))
print("DEBUG MODEL_NAME env =", MODEL_NAME)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(prompt: str, system: str = None):
    model = genai.GenerativeModel(MODEL_NAME, system_instruction=system or "")
    resp = model.generate_content(prompt)
    return resp.text.strip()