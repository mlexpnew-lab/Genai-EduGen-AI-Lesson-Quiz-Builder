from .llm import ask_gemini
from .rag import retrieve

SYS_LESSON = "You are a master teacher. Produce precise, structured outputs grounded strictly in the provided context. If unsure, say so."

def _format_context(docs):
    return "\n\n".join([f"[CTX {i+1}] {d}" for i,d in enumerate(docs)])

def generate_outline(topic: str, question: str = None):
    q = question or f"Create a 5-part lesson outline for {topic}."
    docs,_ = retrieve(q, k=6)
    prompt = f"""{_format_context(docs)}
---
Task: {q}
Output sections:
1) Learning Objectives
2) Key Concepts
3) Teaching Plan (steps)
4) Examples/Exercises
5) Summary & Next Steps
Use citations like (CTX 1/2/3...)."""
    return ask_gemini(prompt, system=SYS_LESSON)

def generate_quiz(topic: str):
    q = f"Create 6 mixed-format questions (MCQ & short) for {topic} with answer key."
    docs,_ = retrieve(q, k=6)
    prompt = f"""{_format_context(docs)}
---
Task: {q}
Format:
Q1) ...
A1) ...
Ensure alignment with objectives; include difficulty tags (Easy/Med/Hard)."""
    return ask_gemini(prompt, system=SYS_LESSON)
