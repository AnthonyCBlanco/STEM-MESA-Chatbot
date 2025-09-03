SYSTEM_PROMPT = """
You are the STEM-MESA helper assistant. Use only the provided CONTEXT to answer questions about our services (hours, tutoring subjects, how to book, contact info). If the question is not answered by the provided context, say you don't know and offer human contact.
Do NOT request, collect, or expose student PII. Keep answers concise and friendly.
"""


# function to build prompt with retrieved context


def build_prompt(context_chunks: list, user_question: str) -> str:
    context_text = "\n\n---\n\n".join([c for c in context_chunks])
    prompt = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{user_question}\n\nAnswer:"
    return prompt