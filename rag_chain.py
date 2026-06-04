import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
from prompts import SYSTEM_PROMPT, HUMAN_PROMPT, format_chunks

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

    # MMR keeps results diverse — without it we often get 4 chunks saying the same thing
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )


def build_chain():
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    retriever = build_retriever()

    # window of 5 keeps recent context for follow-ups without ballooning token usage
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

    # per-document label injected by LangChain before the chunks reach the prompt
    doc_prompt = PromptTemplate(
        input_variables=["page_content", "page_number", "section"],
        template="[Page {page_number} | {section}]\n{page_content}"
    )

    # instructs the condense LLM to resolve pronouns and short follow-ups using chat history
    # without this, questions like "does it require wifi" lose their referent entirely
    condense_prompt = PromptTemplate.from_template(
        """Given the conversation history and a follow-up question, rewrite the follow-up as a standalone question that fully captures the topic being discussed. Always resolve pronouns like 'it', 'this', 'that', 'they' using the conversation history. If the follow-up is already a standalone question, return it unchanged.

Chat history:
{chat_history}

Follow-up question: {question}

Standalone question:"""
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=condense_prompt,
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