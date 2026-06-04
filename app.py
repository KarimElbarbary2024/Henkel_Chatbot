import chainlit as cl
from rag_chain import build_chain


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="How do I turn on Do Not Disturb?",
            message="How do I turn on Do Not Disturb?",
        ),
        cl.Starter(
            label="How do I set up iCloud?",
            message="How do I set up iCloud?",
        ),
        cl.Starter(
            label="How do I take a photo?",
            message="How do I take a photo?",
        ),
        cl.Starter(
            label="How do I restart my iPhone?",
            message="How do I restart my iPhone?",
        ),
    ]


@cl.on_chat_start
async def start():
    chain = build_chain()
    cl.user_session.set("chain", chain)

    await cl.Message(
        content="Hey, I'm Siri-ously 👋 Ask me anything about the iPhone User Guide for iOS 7.1."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    # handle basic conversational messages before hitting the RAG chain
    greetings = ["hi", "hello", "hey", "how are you", "what's up", "sup", "good morning", "good evening"]
    if message.content.strip().lower() in greetings:
        await cl.Message(
            content="Hey there! I'm Siri-ously doing great :) I'm here to answer your questions about the iPhone User Guide for iOS 7.1. What would you like to know?"
        ).send()
        return

    msg = cl.Message(content="")

    result = None
    async for chunk in chain.astream(
        {"question": message.content},
        config={"callbacks": [cl.AsyncLangchainCallbackHandler(stream_final_answer=True)]}
    ):
        if "answer" in chunk:
            await msg.stream_token(chunk["answer"])
        if chunk:
            result = chunk

    await msg.send()