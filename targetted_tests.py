import os
from dotenv import load_dotenv

load_dotenv()

from rag_chain import build_chain

TARGETED_TESTS = [
    {
        "label": "AirDrop distance",
        "turns": [
            "How do I use AirDrop?",
            "How close do I need to be to someone for it to work?",
        ],
    },
    {
        "label": "iCloud storage management",
        "turns": [
            "How do I set up iCloud?",
            "How do I manage my iCloud storage?",
            "How do I upgrade my iCloud storage plan?",
            "What happens if my iCloud storage runs out?",
        ],
    },
    {
        "label": "Siri making calls",
        "turns": [
            "What can Siri do?",
            "Can I use Siri to make a phone call?",
            "How do I do that?",
        ],
    },
    {
        "label": "AirDrop distance — direct",
        "turns": [
            "What is the maximum distance for AirDrop to work?",
        ],
    },
    {
        "label": "iCloud storage — direct",
        "turns": [
            "How do I buy more iCloud storage?",
        ],
    },
    {
        "label": "Siri calls — direct",
        "turns": [
            "Can Siri make phone calls for me?",
        ],
    },
]


def run():
    chain = build_chain()

    out = open("targeted_test_results.txt", "w", encoding="utf-8")

    def p(*args):
        line = " ".join(str(a) for a in args)
        print(line)
        out.write(line + "\n")

    p("=" * 60)
    p("Siri-ously — Targeted Regression Tests")
    p("=" * 60)
    p()

    for session in TARGETED_TESTS:
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
        p()

    p("=" * 60)
    p("end of results")
    p("=" * 60)

    out.close()
    print("done — results written to targeted_test_results.txt")


if __name__ == "__main__":
    run()