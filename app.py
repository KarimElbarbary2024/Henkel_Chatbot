import chainlit as cl
from rag_chain import build_chain



@cl.on_chat_start
async def start():
    # build once per session and cache it, no reason to rebuild on every message
    chain = build_chain()
    cl.user_session.set("chain", chain)

    await cl.Message(
        content="Hey, I'm Siri-ously 👋 Ask me anything about the iPhone User Guide for iOS 7.1."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    msg = cl.Message(content="")

    result = None
    async for chunk in chain.astream(
        {"question": message.content},
        config={"callbacks": [cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]}
    ):
        if "answer" in chunk:
            await msg.stream_token(chunk["answer"])
        # captures the full result so we can pull source docs after streaming
        if chunk:
            result = chunk

    source_elements = []
    seen_pages = []

    # pull source docs directly from the chain result instead of a second search
    if result and "source_documents" in result:
        for doc in result["source_documents"]:
            page = doc.metadata.get("page_number", "unknown")
            if page not in seen_pages:
                seen_pages.append(page)
                source_elements.append(
                    cl.Text(
                        name=f"Page {page}",
                        content=doc.page_content,
                        display="inline"
                    )
                )

    msg.elements = source_elements
    await msg.send()