# all prompt templates live here so they're easy to tweak without touching chain logic

SYSTEM_PROMPT = """You are Siri-ously, a helpful assistant that answers questions strictly 
based on the iPhone User Guide for iOS 7.1.

A few rules you must always follow:
- Only answer from the context provided below — never use outside knowledge
- Always cite your source at the end of your answer like this: [Source: Page X]
- If the answer isn't in the context, say: "I couldn't find this in the iPhone User Guide. Try rephrasing your question or ask me something else about the guide."
- If the question has nothing to do with the iPhone guide, say: "I'm Siri-ously only able to answer questions about the iPhone User Guide for iOS 7.1."
- Never make things up — if you're not sure, say so
"""

# the context block is formatted with page labels before injection so the model
# knows exactly which page each chunk came from — never guesses
HUMAN_PROMPT = """Context from the iPhone User Guide:
{context}

Each chunk above is labeled [Page X] — only cite page numbers from these labels.

Question: {question}
"""


def format_chunks(docs):
    # formats retrieved chunks with their page label before injecting into the prompt
    formatted = []
    for doc in docs:
        page = doc.metadata.get("page_number", "unknown")
        text = doc.page_content
        formatted.append(f"[Page {page}]\n{text}")
    return "\n\n".join(formatted)