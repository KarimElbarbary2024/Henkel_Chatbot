import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from prompts import SYSTEM_PROMPT, HUMAN_PROMPT

load_dotenv()

COLLECTION_NAME = "iphone-user-guide"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def build_retriever():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # MMR algorithm keeps results diverse across the document. Without it, we often get 4 chunks saying essentially the same thing
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )


def build_chain():
    from langchain.prompts import PromptTemplate
    from prompts import SYSTEM_PROMPT, HUMAN_PROMPT, format_chunks

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    retriever = build_retriever()

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
    ])

    doc_prompt = PromptTemplate(
        input_variables=["page_content", "page_number", "section"],
        template="[Page {page_number} | {section}]\n{page_content}"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt,
            "document_prompt": doc_prompt,
            "document_separator": "\n\n",
        },
        return_source_documents=True,
        condense_question_llm=llm,
        output_key="answer"
    )

    return chain