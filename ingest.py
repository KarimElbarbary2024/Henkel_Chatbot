import os
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

PDF_PATH = "data/iphone_user_guide.pdf"
SOURCE_NAME = "iphone_user_guide.pdf"
COLLECTION_NAME = "iphone-user-guide"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def load_pdf(path):
    # page numbers are stored here so we can cite them later in the response
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        # some pages are just images or diagrams with no text layer, skip them
        if text and text.strip():
            pages.append({
                "text": text.strip(),
                "page_number": i + 1
            })
    print("pages loaded:", len(pages))
    return pages


def chunk_pages(pages):
    # separators order matters — tries paragraph breaks before falling back to sentences
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    for page in pages:
        chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(chunks):
            # chunk_index resets per page, not globally
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "page_number": page["page_number"],
                    "source": SOURCE_NAME,
                    "chunk_index": i
                }
            })

    print("chunks created:", len(all_chunks))
    return all_chunks


def setup_collection(client, dim=1536):
    # skip creation if collection already exists
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print("collection already exists, skipping")
        return
    # 1536 matches the output dimension of text-embedding-3-small
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )
    print("collection created")


def ingest():
    pages = load_pdf(PDF_PATH)
    chunks = chunk_pages(pages)

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    setup_collection(client)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # separate texts and metadata before uploading — from_texts expects them split
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # force_recreate=False makes sure we don't wipe the collection if run twice
    QdrantVectorStore.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        force_recreate=False
    )

    print("ingestion complete,", len(chunks), "chunks uploaded")


if __name__ == "__main__":
    ingest()