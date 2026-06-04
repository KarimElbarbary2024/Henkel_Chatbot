# Henkel RAG Chatbot (Siri-ously)

Chatbot that answers questions about the iPhone User Guide for iOS 7.1. Answers are pulled strictly from the document and every response cites the page number.

Stack: LangChain, Qdrant Cloud, Chainlit, Docker.

Embeddings: text-embedding-3-small — Chat model: gpt-4o-mini

## How to run

Fill in your keys by copying .env.example to .env, then run:

docker build -t chatbot:1.0 .

docker run -p 8000:8000 --env-file .env chatbot:1.0

App runs on port 8000, open localhost:8000 in your browser.

No ingestion step needed, the Qdrant index is already populated.

## Keys needed

OPENAI_API_KEY — platform.openai.com/api-keys

QDRANT_URL and QDRANT_API_KEY — cloud.qdrant.io

ingest.py is included for reference only.