import os
import re
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()

COLLECTION_NAME = "iphone-user-guide"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------------------------------------------------------------------------
# test definitions (highly recommend checking out btw)
# each test has:
#   query        - what gets embedded and sent to the retriever
#   expect_pages - at least one of these printed page numbers must appear in results
#   expect_sections - at least one of these section names must appear (substring match)
#   should_miss  - if True, I EXPECT zero relevant results (off-topic / hallucination trap)
#   label        - test name
# ---------------------------------------------------------------------------

TESTS = [
    # --- core retrieval: direct section name queries ---
    {
        "label": "Direct: Do Not Disturb",
        "query": "Do Not Disturb",
        "expect_pages": [32],
        "expect_sections": ["Do Not Disturb"],
    },
    {
        "label": "Direct: AirDrop",
        "query": "AirDrop",
        "expect_pages": [32],
        "expect_sections": ["AirDrop"],
    },
    {
        "label": "Direct: iCloud",
        "query": "iCloud",
        "expect_pages": [17],
        "expect_sections": ["iCloud"],
    },
    {
        "label": "Direct: VoiceOver",
        "query": "VoiceOver",
        "expect_pages": [127],
        "expect_sections": ["VoiceOver"],
    },
    {
        "label": "Direct: Safari",
        "query": "Safari",
        "expect_pages": [55, 56, 57, 58, 59],
        "expect_sections": ["Safari"],
    },

    # --- natural language paraphrases (user won't say section names) ---
    {
        "label": "NL: how to silence phone",
        "query": "how do I silence my phone or turn off sounds",
        "expect_pages": [32],
        "expect_sections": ["Sounds and silence", "Do Not Disturb"],
    },
    {
        "label": "NL: how to make a phone call",
        "query": "how do I make a phone call",
        "expect_pages": [44],
        "expect_sections": ["Phone calls"],
    },
    {
        "label": "NL: set up email account",
        "query": "how do I set up my email account",
        "expect_pages": [16, 54],
        "expect_sections": ["Set up mail and other accounts", "Mail settings"],
    },
    {
        "label": "NL: connect to wifi",
        "query": "how do I connect to a wireless network",
        "expect_pages": [15],
        "expect_sections": ["Connect to Wi-Fi"],
    },
    {
        "label": "NL: take a photo",
        "query": "how do I take a picture with my iPhone",
        "expect_pages": [79, 78],
        "expect_sections": ["Take photos and videos", "Camera at a glance"],
    },
    {
        "label": "NL: share photos",
        "query": "how do I share a photo with someone",
        "expect_pages": [76, 80],
        "expect_sections": ["Share photos and videos", "View, share, and print"],
    },
    {
        "label": "NL: turn on accessibility",
        "query": "how do I enable accessibility features",
        "expect_pages": [126, 127],
        "expect_sections": ["Accessibility features", "Accessibility Shortcut"],
    },
    {
        "label": "NL: forgot passcode",
        "query": "I forgot my passcode how do I unlock my iPhone",
        "expect_pages": [37, 154],
        "expect_sections": ["Security", "Restart or reset iPhone"],
    },
    {
        "label": "NL: battery life",
        "query": "how do I check my battery or make it last longer",
        "expect_pages": [39],
        "expect_sections": ["Charge and monitor the battery"],
    },
    {
        "label": "NL: backup iPhone",
        "query": "how do I back up my iPhone",
        "expect_pages": [156],
        "expect_sections": ["Back up iPhone"],
    },
    {
        "label": "NL: send a text message",
        "query": "how do I send a text message",
        "expect_pages": [67],
        "expect_sections": ["Send and receive messages", "SMS, MMS, and iMessages"],
    },
    {
        "label": "NL: video call",
        "query": "how do I make a video call",
        "expect_pages": [112, 113],
        "expect_sections": ["FaceTime at a glance", "Make and answer calls"],
    },
    {
        "label": "NL: get directions",
        "query": "how do I get directions to somewhere",
        "expect_pages": [87],
        "expect_sections": ["Get directions"],
    },
    {
        "label": "NL: download an app",
        "query": "how do I download a new app",
        "expect_pages": [103, 104],
        "expect_sections": ["App Store at a glance", "Purchase, redeem, and download"],
    },

    # --- duplicate heading disambiguation ---
    # these headings appear in two chapters; we want the RIGHT one (not a deal breaker)
    {
        "label": "Duplicate: Voice Control (main features, not Accessibility)",
        "query": "how do I use voice control to dial a number",
        "expect_pages": [29],
        "expect_sections": ["Voice Control"],
    },
    {
        "label": "Duplicate: Visual voicemail (Phone chapter)",
        "query": "how do I listen to my voicemail messages",
        "expect_pages": [47],
        "expect_sections": ["Visual voicemail"],
    },
    {
        "label": "Duplicate: Control playback (Music/Videos, not Podcasts)",
        "query": "how do I pause or skip a song",
        "expect_pages": [61, 90],
        "expect_sections": ["Browse and play", "Control playback"],
    },
    {
        "label": "Duplicate: Contacts (Phone chapter vs Contacts app)",
        "query": "how do I add a new contact to my phone",
        "expect_pages": [115, 48],
        "expect_sections": ["Add contacts", "Contacts"],
    },
    {
        "label": "Duplicate: Siri (main chapter vs Accessibility)",
        "query": "how do I ask Siri a question",
        "expect_pages": [41, 42],
        "expect_sections": ["Make requests", "Tell Siri about yourself"],
    },

    # --- early pages (p8-p9, the ones that were missed before) ---
    {
        "label": "Early page: iPhone overview (p8)",
        "query": "overview of the iPhone hardware and what comes in the box",
        "expect_pages": [8, 9],
        "expect_sections": ["iPhone overview", "Accessories"],
    },
    {
        "label": "Early page: Accessories (p9)",
        "query": "what accessories come with the iPhone",
        "expect_pages": [9],
        "expect_sections": ["Accessories"],
    },

    # --- multi-word / punctuation heavy section names ---
    {
        "label": "Punctuation: AirDrop iCloud and other ways to share",
        "query": "what are the different ways I can share things from my iPhone",
        "expect_pages": [32, 33],
        "expect_sections": ["AirDrop, iCloud, and other ways to share", "Transfer files"],
    },
    {
        "label": "Punctuation: Call forwarding caller ID",
        "query": "how do I turn on call forwarding or hide my number",
        "expect_pages": [48],
        "expect_sections": ["Call forwarding, call waiting, and caller ID"],
    },
    {
        "label": "Em dash: Genius made for you",
        "query": "what is the Genius feature in music",
        "expect_pages": [64],
        "expect_sections": ["Genius—made for you"],
    },
    {
        "label": "Punctuation: Sell or give away iPhone?",
        "query": "how do I wipe my iPhone before selling it",
        "expect_pages": [158],
        "expect_sections": ["Sell or give away iPhone?"],
    },

    # --- sparse / short sections that might have thin embeddings (very subtle but important) ---
    {
        "label": "Short section: HDR",
        "query": "what is HDR in the iPhone camera",
        "expect_pages": [80],
        "expect_sections": ["HDR"],
    },
    {
        "label": "Short section: On the level",
        "query": "how do I use the compass as a level",
        "expect_pages": [109],
        "expect_sections": ["On the level"],
    },
    {
        "label": "Short section: AirPrint",
        "query": "how do I print something from my iPhone",
        "expect_pages": [34],
        "expect_sections": ["AirPrint"],
    },

    # --- settings pages (tend to be thin boilerplate) ---
    {
        "label": "Settings: Safari settings",
        "query": "where do I change Safari browser settings",
        "expect_pages": [59],
        "expect_sections": ["Safari settings"],
    },
    {
        "label": "Settings: Music settings",
        "query": "how do I change equalizer or sound check settings for music",
        "expect_pages": [66],
        "expect_sections": ["Music settings"],
    },
    {
        "label": "Settings: Camera settings",
        "query": "how do I change camera grid or location settings",
        "expect_pages": [81],
        "expect_sections": ["Camera settings"],
    },

    # --- off-topic: should return results but none should be relevant ---
    # mark these should_miss=True and check that NO result page is in a
    # reasonable "expected" range — I just verify the query doesn't hallucinate
    # a confident hit on something totally unrelated
    {
        "label": "Off-topic: capital of France",
        "query": "what is the capital of France",
        "should_miss": True,
        "expect_pages": [],
        "expect_sections": [],
    },
    {
        "label": "Off-topic: who is Lionel Messi",
        "query": "who is Lionel Messi",
        "should_miss": True,
        "expect_pages": [],
        "expect_sections": [],
    },
    {
        "label": "Off-topic: battery capacity spec",
        "query": "what is the battery capacity mAh of the iPhone 5",
        "should_miss": True,
        "expect_pages": [],
        "expect_sections": [],
    },
    {
        "label": "Off-topic: latest iPhone model",
        "query": "what is the latest iPhone model available right now",
        "should_miss": True,
        "expect_pages": [],
        "expect_sections": [],
    },

    # --- adversarial: queries that sound on-topic but test wrong-chapter pulls (a good practice to make it silly-user-proof) ---
    {
        "label": "Adversarial: Siri in Accessibility should not beat main Siri chapter",
        "query": "how do I set up Siri and what can it do",
        "expect_pages": [41, 42, 43],
        "expect_sections": ["Make requests", "Tell Siri about yourself", "Siri settings"],
    },
    {
        "label": "Adversarial: VoiceOver accessibility deep feature",
        "query": "how do I use VoiceOver gestures to navigate",
        "expect_pages": [127],
        "expect_sections": ["VoiceOver"],
    },
    {
        "label": "Adversarial: iCloud storage not hardware specs",
        "query": "how do I manage iCloud storage",
        "expect_pages": [17],
        "expect_sections": ["iCloud"],
    },

    # --- typo / informal phrasing ---
    {
        "label": "Typo-style: airdrop",
        "query": "airdrop files to another iphone",
        "expect_pages": [32],
        "expect_sections": ["AirDrop"],
    },
    {
        "label": "Informal: restart frozen iphone",
        "query": "my iphone is frozen how do I restart it",
        "expect_pages": [154],
        "expect_sections": ["Restart or reset iPhone"],
    },
    {
        "label": "Informal: phone keeps ringing how to vibrate only",
        "query": "how do I make my phone vibrate only and stop ringing",
        "expect_pages": [32, 48],
        "expect_sections": ["Sounds and silence", "Ringtones and vibrations"],
    },
]


# ---------------------------------------------------------------------------
# retriever setup
# ---------------------------------------------------------------------------

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
    return store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

OFF_TOPIC_PAGES = {8, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 28,
                   29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 43, 44,
                   47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
                   63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
                   78, 79, 80, 81, 86, 87, 88, 89, 90, 91, 92, 93, 95, 98, 99,
                   100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                   112, 113, 114, 115, 116, 118, 119, 120, 121, 122, 123, 124,
                   125, 126, 127, 137, 138, 139, 140, 141, 142, 145, 146, 147,
                   148, 149, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161}


def run_test(retriever, test):
    docs = retriever.invoke(test["query"])
    result_pages = [d.metadata.get("page_number") for d in docs]
    result_sections = [d.metadata.get("section", "") for d in docs]

    if test.get("should_miss"):
        # for off-topic queries we just report what came back — no pass/fail
        # since the retriever will always return something; the LLM decides relevance
        # we flag it as a warning if results look suspiciously on-topic
        suspicious = [p for p in result_pages if p in OFF_TOPIC_PAGES]
        status = "WARN" if suspicious else "NOTE"
        detail = "retriever returned: pages " + str(result_pages) + " sections " + str(result_sections)
        return status, detail

    page_hit = any(p in test["expect_pages"] for p in result_pages)
    section_hit = any(
        any(s.lower() in rs.lower() for rs in result_sections)
        for s in test["expect_sections"]
    )

    if page_hit and section_hit:
        status = "PASS"
        detail = "pages " + str(result_pages) + " | sections " + str(result_sections)
    elif page_hit:
        status = "PARTIAL"
        detail = "page hit but section mismatch | pages " + str(result_pages) + " | sections " + str(result_sections)
    elif section_hit:
        status = "PARTIAL"
        detail = "section hit but page mismatch | pages " + str(result_pages) + " | sections " + str(result_sections)
    else:
        status = "FAIL"
        detail = "got pages " + str(result_pages) + " | sections " + str(result_sections)
        detail += " | expected pages " + str(test["expect_pages"]) + " or sections " + str(test["expect_sections"])

    return status, detail


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("building retriever...")
    retriever = build_retriever()
    print("retriever ready\n")

    results = {"PASS": 0, "PARTIAL": 0, "FAIL": 0, "WARN": 0, "NOTE": 0}
    failures = []

    for i, test in enumerate(TESTS):
        status, detail = run_test(retriever, test)
        results[status] += 1
        marker = "✓" if status == "PASS" else ("~" if status == "PARTIAL" else ("?" if status in ("WARN", "NOTE") else "✗"))
        print(marker + " [" + status + "] " + test["label"])
        if status != "PASS":
            print("  " + detail)
        if status in ("FAIL", "PARTIAL"):
            failures.append(test["label"])

    total = len(TESTS)
    off_topic_count = sum(1 for t in TESTS if t.get("should_miss"))
    scored = total - off_topic_count

    print("\n--- summary ---")
    print("total tests:", total, "| scored:", scored, "| off-topic (unscored):", off_topic_count)
    print("PASS:", results["PASS"], "| PARTIAL:", results["PARTIAL"], "| FAIL:", results["FAIL"])
    print("score:", str(results["PASS"]) + "/" + str(scored),
          "(" + str(round((results["PASS"] / scored) * 100)) + "%)")

    if failures:
        print("\nneeds attention:")
        for f in failures:
            print("  -", f)