import os
import sys
from dotenv import load_dotenv

load_dotenv()

from rag_chain import build_chain

TEST_CASES = [
    "How do I turn on Do Not Disturb?",
    "How do I set up iCloud?",
    "Tell me about Safari",
    "How do I use AirDrop?",
    "How do I make a phone call?",
    "How do I take a photo?",
    "My iPhone is frozen, what do I do?",
    "How do I send a text message?",
    "How do I connect to Wi-Fi?",
    "How do I back up my iPhone?",
    "How do I use Siri?",
    "How do I set a passcode?",
    "How do I turn on Bluetooth?",
    "How do I share a photo?",
    "What is the capital of France?",
    "Who is Lionel Messi?",
    "What is the battery capacity of the iPhone 5?",
    "What is the latest iPhone model?",
]

MULTI_TURN_SESSIONS = [
    {
        "label": "iCloud memory chain",
        "turns": [
            "How do I set up iCloud?",
            "What can I store in it?",
            "How do I manage the storage?",
            "How do I upgrade my storage plan?",
            "What happens if it runs out?",
        ],
    },
    {
        "label": "Camera and photos chain",
        "turns": [
            "How do I take a photo?",
            "Can I edit it afterwards?",
            "How do I share it with someone?",
            "What about printing it?",
        ],
    },
    {
        "label": "Do Not Disturb and sounds chain",
        "turns": [
            "How do I silence my iPhone?",
            "What is Do Not Disturb?",
            "Can I schedule it?",
            "Will alarms still go off when it's on?",
        ],
    },
    {
        "label": "Siri chain",
        "turns": [
            "What can Siri do?",
            "Can I use Siri to make a phone call?",
            "How do I activate Siri?",
            "What if it doesn't understand me?",
        ],
    },
    {
        "label": "AirDrop chain",
        "turns": [
            "How do I use AirDrop?",
            "How close do I need to be to someone for it to work?",
            "Can I use it to send videos too?",
            "Does it need Wi-Fi?",
        ],
    },
    {
        "label": "Off-topic then back on topic",
        "turns": [
            "Who won the World Cup in 2022?",
            "How do I use AirDrop?",
            "Can I use it to send videos too?",
            "How close do I need to be to someone for it to work?",
        ],
    },
]
def run_tests():
    chain = build_chain()

    out = open("llm_test_results.txt", "w", encoding="utf-8")

    def p(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        out.write(line + "\n")

    p("=" * 60)
    p("Siri-ously — LLM Generation Test Results")
    p("=" * 60)
    p()

    p("--- Single-turn questions ---")
    p()
    for question in TEST_CASES:
        p("Q:", question)
        result = chain.invoke({"question": question, "chat_history": []})
        answer = result.get("answer", result) if isinstance(result, dict) else result
        p("A:", answer)
        p()

    p()
    p("--- Multi-turn conversation sessions ---")

    for session in MULTI_TURN_SESSIONS:
        p()
        p("Session:", session["label"])
        p("-" * 60)
        history = []
        for question in session["turns"]:
            p("Q:", question)
            result = chain.invoke({"question": question, "chat_history": history})
            answer = result.get("answer", result) if isinstance(result, dict) else result
            p("A:", answer)
            p()
            history.append((question, answer))

    p("=" * 60)
    p("end of results")
    p("=" * 60)

    out.close()
    print("done — results written to llm_test_results.txt")


if __name__ == "__main__":
    run_tests()